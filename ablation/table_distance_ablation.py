from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


DATASETS = [
    "sciq",
    "nq",
    "common",
    "simple_questions_wiki",
    "truthfulQA",
    "gsm8k",
]

MODELS = [
    "Gemma-3-4B-it",
    "Llama-3.2-3B-Instruct",
    "Llama-3-8B-Instruct",
]

DISTANCES = [
    "knn",
    "shrinkage_mahalanobis",
    "diagonal_mahalanobis",
    "standardized_euclidean",
]

RESULTS_ROOT = Path("results")


def load_scores(model, dataset, distance):
    test_dir = RESULTS_ROOT / model / dataset / "test"
    label_path = test_dir / "label.npy"
    prediction_path = test_dir / (
        "proposed_"
        f"{distance}.npy"
    )

    if not label_path.exists() or not prediction_path.exists():
        return None, label_path, prediction_path

    labels = np.asarray(np.load(label_path)).reshape(-1)
    predictions = np.asarray(np.load(prediction_path)).reshape(-1)

    if len(labels) != len(predictions):
        raise ValueError(
            f"Length mismatch for {model}/{dataset}/{distance}: "
            f"{len(labels)} labels versus {len(predictions)} predictions."
        )

    if not np.all(np.isfinite(predictions)):
        raise ValueError(
            f"Non-finite predictions found for "
            f"{model}/{dataset}/{distance}."
        )

    unique_labels = np.unique(labels)
    if len(unique_labels) != 2:
        raise ValueError(
            f"Expected two label classes for {model}/{dataset}, "
            f"found {unique_labels}."
        )

    scores = {
        "dataset": dataset,
        "model": model,
        "distance": distance,
        "n_test": len(labels),
        "auroc": roc_auc_score(labels, predictions),
        "auprc": average_precision_score(labels, predictions),
    }

    return scores, label_path, prediction_path


def format_metric(auroc, auprc):
    return f"{auroc:.4f}/{auprc:.4f}"


def main():
    rows = []
    missing = []

    for dataset in DATASETS:
        for model in MODELS:
            for distance in DISTANCES:
                scores, label_path, prediction_path = load_scores(
                    model,
                    dataset,
                    distance,
                )

                if scores is None:
                    missing.append(
                        {
                            "dataset": dataset,
                            "model": model,
                            "distance": distance,
                            "missing_label": not label_path.exists(),
                            "missing_prediction": (
                                not prediction_path.exists()
                            ),
                        }
                    )
                    continue

                rows.append(scores)

    if not rows:
        print(f"No result files found under {RESULTS_ROOT.resolve()}")
        return

    results = pd.DataFrame(rows)

    results["AUROC/AUPRC"] = results.apply(
        lambda row: format_metric(
            row["auroc"],
            row["auprc"],
        ),
        axis=1,
    )

    # Print one model-by-distance table for each dataset.
    for dataset in DATASETS:
        dataset_results = results[
            results["dataset"] == dataset
        ]

        if dataset_results.empty:
            continue

        table = (
            dataset_results.pivot(
                index="model",
                columns="distance",
                values="AUROC/AUPRC",
            )
            .reindex(
                index=MODELS,
                columns=DISTANCES,
            )
        )

        table.index.name = "model"
        table.columns.name = "distance"

        print(f"\n{dataset}: AUROC/AUPRC")
        print(table.to_string(na_rep="-"))

    # Print a model-by-distance table averaged across datasets.
    mean_results = (
        results.groupby(
            ["model", "distance"],
            as_index=False,
        )[["auroc", "auprc"]]
        .mean()
    )

    mean_results["AUROC/AUPRC"] = mean_results.apply(
        lambda row: format_metric(
            row["auroc"],
            row["auprc"],
        ),
        axis=1,
    )

    mean_table = (
        mean_results.pivot(
            index="model",
            columns="distance",
            values="AUROC/AUPRC",
        )
        .reindex(
            index=MODELS,
            columns=DISTANCES,
        )
    )

    mean_table.index.name = "model"
    mean_table.columns.name = "distance"

    print("\nMean across available datasets: AUROC/AUPRC")
    print(mean_table.to_string(na_rep="-"))

    if missing:
        missing_frame = pd.DataFrame(missing)
        total = len(DATASETS) * len(MODELS) * len(DISTANCES)

        print(
            f"\nMissing {len(missing_frame)} "
            f"of {total} combinations:"
        )
        print(missing_frame.to_string(index=False))


if __name__ == "__main__":
    main()