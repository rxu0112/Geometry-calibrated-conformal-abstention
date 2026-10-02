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
    "Llama-3-8B-Instruct",
    "Llama-3.2-3B-Instruct",
]

RESULTS_ROOT = Path("results")


def load_array(path):
    return np.asarray(np.load(path)).reshape(-1)


def compute_scores(labels, predictions):
    return {
        "auroc": roc_auc_score(labels, predictions),
        "auprc": average_precision_score(labels, predictions),
    }


def main():
    rows = []
    missing = []

    for model in MODELS:
        for dataset in DATASETS:
            test_dir = RESULTS_ROOT / model / dataset / "test"
            paths = {
                "label": test_dir / "label.npy",
                "label_pca": test_dir / "label_pca.npy", # label_conf_only.npy
                "ours": test_dir / f"proposed.npy",
                "pca": test_dir / f"proposed_pca.npy", # proposed_conf_only.npy
            }

            missing_names = [name for name, path in paths.items() if not path.exists()]
            if missing_names:
                missing.append(f"{model}/{dataset}: {', '.join(missing_names)}")
                continue

            labels = load_array(paths["label"])
            labels_pca = load_array(paths["label_pca"])

            ours = load_array(paths["ours"])
            pca = load_array(paths["pca"])

            ours_scores = compute_scores(labels, ours)
            pca_scores = compute_scores(labels_pca, pca)
            rows.append(
                {
                    "Model": model,
                    "Dataset": dataset,
                    "Ours AUROC": ours_scores["auroc"],
                    "PCA AUROC": pca_scores["auroc"],
                    "Delta AUROC": ours_scores["auroc"] - pca_scores["auroc"],
                    "Ours AUPRC": ours_scores["auprc"],
                    "PCA AUPRC": pca_scores["auprc"],
                    "Delta AUPRC": ours_scores["auprc"] - pca_scores["auprc"],
                }
            )


    if not rows:
        print(f"No complete result combinations found under {RESULTS_ROOT.resolve()}")
    else:
        results = pd.DataFrame(rows)
        numeric_columns = results.columns.difference(["Model", "Dataset"])

        print("\nPer model-dataset comparison (Delta = Ours - PCA)")
        print(results.round(4).to_string(index=False))

        print("\nMacro average over available configurations")
        print(results[numeric_columns].mean().round(4).to_string())

        wins = pd.Series(
            {
                "AUROC wins": int((results["Delta AUROC"] > 0).sum()),
                "AUROC ties": int(np.isclose(results["Delta AUROC"], 0).sum()),
                "AUPRC wins": int((results["Delta AUPRC"] > 0).sum()),
                "AUPRC ties": int(np.isclose(results["Delta AUPRC"], 0).sum()),
                "Configurations": len(results),
            }
        )
        print("\nWin counts")
        print(wins.to_string())

        print("\nAverage over datasets for each model")
        model_summary = (
            results.groupby("Model", sort=False)[
                [
                    "Ours AUROC",
                    "PCA AUROC",
                    "Delta AUROC",
                    "Ours AUPRC",
                    "PCA AUPRC",
                    "Delta AUPRC",
                ]
            ]
            .mean()
            .reindex(MODELS)
            .round(4)
        )

        print(model_summary.to_string())
    if missing:
        print(f"\nMissing {len(missing)} configurations")
        for item in missing:
            print(f"  {item}")


if __name__ == "__main__":
    main()
