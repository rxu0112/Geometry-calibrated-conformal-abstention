import pickle
from pathlib import Path

import numpy as np
import pandas as pd


GENERALIZATION_RESULT_PATH = Path("results/calibrator_generalization_ablation.pkl")
SOURCE = "simple_questions_wiki"
TARGETS = ["sciq", "nq", "common", "truthfulQA", "gsm8k"]
MODELS = [
    "Llama-3.2-3B-Instruct",
    "Gemma-3-4B-it",
    "Llama-3-8B-Instruct",
]

DISPLAY_NAMES = {
    "sciq": "SciQ",
    "nq": "NQ",
    "common": "CQA",
    "truthfulQA": "TQA",
    "gsm8k": "GSM8K",
}

TARGET_PARTICIPATION = 0.5


def conformal_threshold(scores, participation):
    """Scores are lower-is-better."""
    scores = np.sort(np.asarray(scores).reshape(-1))
    n = len(scores)
    rank = int(np.ceil(participation * (n + 1)))

    if rank > n:
        return np.inf

    return scores[rank - 1]


def compute_eq6_value(calibration_scores, calibration_labels, threshold):
    """Compute the source-calibration Eq. 6 plug-in value."""
    calibration_scores = np.asarray(calibration_scores).reshape(-1)
    calibration_labels = np.asarray(calibration_labels).reshape(-1)

    n = len(calibration_labels)
    correct_scores = calibration_scores[calibration_labels == 1]
    c = len(correct_scores)

    if c == 0:
        return 0.0

    k = np.sum(correct_scores <= threshold)
    correct_retention = k / (c + 1)

    participation_upper = min(
        1.0,
        TARGET_PARTICIPATION + 1.0 / (n + 1),
    )

    return (
        correct_retention
        * (c / n)
        / participation_upper
    )


with GENERALIZATION_RESULT_PATH.open("rb") as file:
    results = pickle.load(file)

rows = []

for model in MODELS:
    # SQW scores are independent of SQW calibrator training because they
    # came from its held-out test split.
    source_record = results[model][SOURCE][SOURCE]

    source_confidence = np.asarray(
        source_record["y_proba_test"]
    ).reshape(-1)
    source_labels = np.asarray(source_record["y_test"]).reshape(-1)

    # Higher confidence is better, so negate it to obtain a
    # lower-is-better conformal score.
    source_scores = -source_confidence

    threshold = conformal_threshold(
        source_scores,
        TARGET_PARTICIPATION,
    )

    eq6_value = compute_eq6_value(
        source_scores,
        source_labels,
        threshold,
    )

    n_source = len(source_labels)
    participation_upper = (
        TARGET_PARTICIPATION + 1.0 / (n_source + 1)
    )

    for target in TARGETS:
        target_record = results[model][SOURCE][target]

        target_scores = -np.asarray(
            target_record["y_proba_test"]
        ).reshape(-1)
        target_labels = np.asarray(
            target_record["y_test"]
        ).reshape(-1)

        retained = target_scores <= threshold
        participation = retained.mean()

        conditional_correctness = (
            target_labels[retained].mean()
            if retained.any()
            else np.nan
        )

        rows.append(
            {
                "Model": model,
                "Test domain": DISPLAY_NAMES[target],
                "Participation": participation,
                "Participation gap": (
                    participation - TARGET_PARTICIPATION
                ),
                "Conditional correctness": conditional_correctness,
                "Source Eq. 6 value": eq6_value,
                "Correctness gap": (
                    conditional_correctness - eq6_value
                ),
            }
        )

table = pd.DataFrame(rows)

for model in MODELS:
    model_table = table[table["Model"] == model].drop(
        columns="Model"
    )

    print(f"\n{model}: calibrated on SQW")
    print(
        model_table.round(4).to_string(index=False)
    )