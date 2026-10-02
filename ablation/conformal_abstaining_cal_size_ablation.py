import os
import pickle

import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm


CALIBRATION_RATIOS = [0.05, 0.25, 0.50, 0.75, 1.00]
PARTICIPATION_LEVELS = np.asarray([0.5])
N_TRIALS = 1000
TOLERANCE = 1e-12


def conformal_thresholds(sorted_scores, participation_levels):
    """Finite-sample conformal thresholds for lower-is-better scores."""
    n = len(sorted_scores)
    if n == 0:
        raise ValueError("The calibration set cannot be empty.")

    thresholds = []
    for level in participation_levels:
        rank = int(np.ceil(level * (n + 1)))
        if rank > n:
            thresholds.append(np.inf)
        else:
            thresholds.append(sorted_scores[max(rank - 1, 0)])
    return np.asarray(thresholds)


def stratified_calibration_subset(temp_indices, labels, ratio, random_state):
    """Subsample the calibration pool while preserving its class proportions."""
    if ratio == 1.0:
        return temp_indices.copy()

    desired_size = int(round(ratio * len(temp_indices)))
    desired_size = max(desired_size, len(np.unique(labels[temp_indices])))
    desired_size = min(desired_size, len(temp_indices) - 1)

    _, calibration_indices = train_test_split(
        temp_indices,
        test_size=desired_size,
        random_state=random_state,
        stratify=labels[temp_indices],
    )
    return calibration_indices


def evaluate_trial(calibration_scores, calibration_labels, test_scores, test_labels):
    """Evaluate all participation levels for one calibration/test trial."""
    n_cal = len(calibration_scores)
    c_cal = int(np.sum(calibration_labels == 1))
    calibration_accuracy = c_cal / n_cal

    sorted_calibration_scores = np.sort(calibration_scores)
    sorted_correct_scores = np.sort(
        calibration_scores[calibration_labels == 1]
    )
    thresholds = conformal_thresholds(
        sorted_calibration_scores, PARTICIPATION_LEVELS
    )

    participation = []
    conditional_correctness = []
    conditional_lower_bounds = []
    correct_retention = []
    correct_retention_lower_bounds = []
    participation_lower_violations = []
    participation_upper_violations = []
    correct_retention_violations = []
    conditional_lower_bound_violations = []

    test_correct_mask = test_labels == 1
    n_test_correct = int(np.sum(test_correct_mask))

    for level, threshold in zip(PARTICIPATION_LEVELS, thresholds):
        retained = test_scores <= threshold
        n_retained = int(np.sum(retained))
        n_retained_correct = int(np.sum(retained & test_correct_mask))

        observed_participation = n_retained / len(test_scores)
        if n_retained == 0:
            observed_conditional_correctness = np.nan
        else:
            observed_conditional_correctness = (
                n_retained_correct / n_retained
            )

        # If k of the c correct calibration scores are at most tau, the exact
        # finite-sample correct-retention coverage is k/(c+1).
        if c_cal == 0:
            correct_retention_bound = 0.0
        else:
            k_correct = int(
                np.searchsorted(
                    sorted_correct_scores, threshold, side="right"
                )
            )
            correct_retention_bound = k_correct / (c_cal + 1)

        participation_upper_bound = min(
            1.0, level + 1.0 / (n_cal + 1)
        )
        conditional_bound = (
            correct_retention_bound
            * calibration_accuracy
            / participation_upper_bound
        )
        conditional_bound = float(np.clip(conditional_bound, 0.0, 1.0))

        if n_test_correct == 0:
            observed_correct_retention = np.nan
        else:
            observed_correct_retention = (
                n_retained_correct / n_test_correct
            )

        participation.append(observed_participation)
        conditional_correctness.append(observed_conditional_correctness)
        conditional_lower_bounds.append(conditional_bound)
        correct_retention.append(observed_correct_retention)
        correct_retention_lower_bounds.append(correct_retention_bound)

        participation_lower_violations.append(
            observed_participation + TOLERANCE < level
        )
        participation_upper_violations.append(
            observed_participation - TOLERANCE > participation_upper_bound
        )
        correct_retention_violations.append(
            np.isfinite(observed_correct_retention)
            and observed_correct_retention + TOLERANCE
            < correct_retention_bound
        )
        conditional_lower_bound_violations.append(
            np.isfinite(observed_conditional_correctness)
            and observed_conditional_correctness + TOLERANCE
            < conditional_bound
        )

    return {
        "n_cal": n_cal,
        "c_cal": c_cal,
        "calibration_accuracy": calibration_accuracy,
        "thresholds": thresholds,
        "participation": np.asarray(participation),
        "conditional_correctness": np.asarray(conditional_correctness),
        "conditional_lower_bounds": np.asarray(conditional_lower_bounds),
        "correct_retention": np.asarray(correct_retention),
        "correct_retention_lower_bounds": np.asarray(
            correct_retention_lower_bounds
        ),
        "participation_lower_violations": np.asarray(
            participation_lower_violations, dtype=bool
        ),
        "participation_upper_violations": np.asarray(
            participation_upper_violations, dtype=bool
        ),
        "correct_retention_violations": np.asarray(
            correct_retention_violations, dtype=bool
        ),
        "conditional_lower_bound_violations": np.asarray(
            conditional_lower_bound_violations, dtype=bool
        ),
    }


def mean_and_std(values, allow_nan=False):
    values = np.asarray(values)
    if allow_nan:
        return (
            np.nanmean(values, axis=0),
            np.nanstd(values, axis=0, ddof=1),
        )
    return (
        np.mean(values, axis=0),
        np.std(values, axis=0, ddof=1),
    )


def summarize_trials(trials, ratio):
    summary = {
        "sampling ratio": ratio,
        "participation levels": PARTICIPATION_LEVELS.copy(),
    }

    scalar_fields = [
        "n_cal",
        "c_cal",
        "calibration_accuracy",
    ]
    vector_fields = [
        "thresholds",
        "participation",
        "conditional_correctness",
        "conditional_lower_bounds",
        "correct_retention",
        "correct_retention_lower_bounds",
    ]

    for field in scalar_fields:
        mean, std = mean_and_std(
            [trial[field] for trial in trials],
            allow_nan=False,
        )
        summary[f"{field} mean"] = mean
        summary[f"{field} std"] = std

    for field in vector_fields:
        mean, std = mean_and_std(
            [trial[field] for trial in trials],
            allow_nan=field in {"conditional_correctness", "correct_retention"},
        )
        summary[f"{field} mean"] = mean
        summary[f"{field} std"] = std

    violation_fields = [
        "participation_lower_violations",
        "participation_upper_violations",
        "correct_retention_violations",
        "conditional_lower_bound_violations",
    ]

    conditional_values = np.stack(
        [trial["conditional_correctness"] for trial in trials]
    )
    conditional_evaluable = np.sum(
        np.isfinite(conditional_values), axis=0
    )

    for field in violation_fields:
        violations = np.stack([trial[field] for trial in trials])
        counts = np.sum(violations, axis=0)
        summary[f"{field} count"] = counts
        if field == "conditional_lower_bound_violations":
            summary[f"{field} rate"] = np.divide(
                counts,
                conditional_evaluable,
                out=np.full_like(counts, np.nan, dtype=float),
                where=conditional_evaluable > 0,
            )
        else:
            summary[f"{field} rate"] = counts / len(trials)

    summary["conditional bound evaluable count"] = conditional_evaluable

    return summary


def main():
    dataset = "simple_questions_wiki"
    model = "Llama-3.2-3B-Instruct"
    test_dir = f"results/{model}/{dataset}/test"
    labels = np.load(f"{test_dir}/label.npy").reshape(-1)
    confidences = np.load(
        f"{test_dir}/proposed.npy"
    ).reshape(-1)

    if len(labels) != len(confidences):
        raise ValueError(
            f"Length mismatch: labels={len(labels)}, "
            f"confidences={len(confidences)}"
        )
    if len(np.unique(labels)) != 2:
        raise ValueError("Both correctness labels are required.")

    comparison = []

    all_indices = np.arange(len(labels))
    for ratio in CALIBRATION_RATIOS:
        print(f"Processing {model} {dataset} ratio={ratio}")
        trials = []

        for random_state in range(1, N_TRIALS + 1):
            temp_indices, test_indices = train_test_split(
                all_indices,
                test_size=0.5,
                random_state=random_state,
                stratify=labels,
            )
            calibration_indices = stratified_calibration_subset(
                temp_indices,
                labels,
                ratio,
                random_state=random_state + 10_000,
            )

            # Higher confidence is better, whereas conformal scores use
            # the convention that lower values are retained.
            calibration_scores = -confidences[calibration_indices]
            test_scores = -confidences[test_indices]

            trials.append(
                evaluate_trial(
                    calibration_scores,
                    labels[calibration_indices],
                    test_scores,
                    labels[test_indices],
                )
            )

        summary = summarize_trials(trials, ratio)
        comparison.append(summary)
        violation_count = int(
            summary[
                "conditional_lower_bound_violations count"
            ][0]
        )
        evaluable_count = int(
            summary["conditional bound evaluable count"][0]
        )
        violation_rate = summary[
            "conditional_lower_bound_violations rate"
        ][0]
        print(
            f"n={summary['n_cal mean']:.1f}+/-"
            f"{summary['n_cal std']:.1f}, "
            f"c={summary['c_cal mean']:.1f}+/-"
            f"{summary['c_cal std']:.1f}, "
            "true participation="
            f"{summary['participation mean'][0]:.4f}+/-"
            f"{summary['participation std'][0]:.4f}, "
            "conditional correctness="
            f"{summary['conditional_correctness mean'][0]:.4f}+/-"
            f"{summary['conditional_correctness std'][0]:.4f}, "
            "certificate="
            f"{summary['conditional_lower_bounds mean'][0]:.4f}+/-"
            f"{summary['conditional_lower_bounds std'][0]:.4f}, "
            "conditional-bound empirical violations="
            f"{violation_count}/{evaluable_count} "
            f"({violation_rate:.2%})"
        )


if __name__ == "__main__":
    main()
