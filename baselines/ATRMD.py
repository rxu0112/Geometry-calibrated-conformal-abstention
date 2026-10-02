#!/usr/bin/env python3
"""Single-file ATRMD baseline with your partner's learned layer combination.

Edit the PATHS section below, then run on your GPU machine:
    python ATRMD.py --model Llama-3.2-3B-Instruct --dataset nq --device cuda:1

The only arguments are --model, --dataset, and --device. The model determines
the layers automatically. Training data builds the reference; only test is scored.
Input paths are printed before the model is loaded.

No run.sh, YAML files, extraction script, or partner Python files are needed.
Python packages ARE still needed: numpy, pandas, torch, transformers, datasets,
scikit-learn, and the
same lm-polygraph fork pinned in your partner's requirements.txt:
    lm-polygraph @ git+https://github.com/ArtemVazh/lm-polygraph.git@38a5202baf613a8b3b058d007d396b121a5bbfb5
This fork supplies the token-embedding statistics used by the original code.

Preserved: token-average relative distance, label threshold 0.3, seed 1,
up to 5,000 training examples, and 1,000 C4 background examples. The layer
combination matches run_polygraph.py's aggregated relative-distance configuration: a 50/50 training
split with random_state=42, raw layer scores, PCA(n_components=10), and
Ridge(positive=False), trained to predict 1 - training_label. Generation uses
the original CSV loader's 100-token cap: the supplied run.sh passes relative
processed_data paths that skip its historical-length alignment. Consequently,
the indices file and historical pickle are not inputs to this version.
This is the aggregated relative-distance variant enabled by run.sh.
Removed: other baselines, hybrid combinations, and evaluation metrics.
Only y_train is needed from the labels file; test labels are not used to score.

Output:
    output_dir/ATRMD.npy  [test examples]
One learned combined score per test example, in the original test CSV row order.
The Python script is ATRMD.py; the saved score array is ATRMD.npy.
"""

import argparse
import logging
import os


parser = argparse.ArgumentParser(description="ATRMD Baseline")
parser.add_argument("--device", default="cuda:0", type=str)
parser.add_argument("--dataset", type=str,
                    choices=["sciq", "simple_questions_wiki", "nq", "truthfulQA",
                             "gsm8k", "common"], required=True)
parser.add_argument("--model", type=str,
                    choices=["Llama-3.2-3B-Instruct", "Gemma-3-4B-it",
                             "Llama-3-8B-Instruct"], required=True)
args = parser.parse_args()


# ============================================================================
# PATHS TO CHECK TOGETHER -- all input/output locations are defined here.
# Use absolute paths so running from a different directory changes nothing.
# ============================================================================

# 1. INPUT: model directory (same as AttentionScore.py).
if args.model  == "Llama-3.2-3B-Instruct":
    model_path = 'meta-llama/Llama-3.2-3B-Instruct'
elif args.model == "Llama-3-8B-Instruct":
    model_path = "meta-llama/Meta-Llama-3-8B-Instruct"
elif args.model == "Gemma-3-4B-it":
    model_path = "google/gemma-3-4b-it"

# 2. INPUT: folder containing nq/, sciq/, truthfulQA/, etc. MUST FILL IN.
train_data_path = f"processed_data/{args.dataset}/{args.model}_train_data.csv"
test_data_path = f"processed_data/{args.dataset}/{args.model}_test_data.csv"

# 3. INPUT: training labels. This file must contain the key "y_train".
labels_path = f"processed_data/{args.dataset}/{args.model}_labels.pt"

# 4. INPUT: C4 background data. Choose ONE way to supply the same C4 shard.
# None = use the Hugging Face dataset/cache below, downloading if not cached.
# For a local shard, set e.g. "/your/data/c4-train.00000-of-01024.json.gz".
C4_LOCAL_FILE = None
C4_DATASET_ID = "allenai/c4"
C4_SHARD = "en/c4-train.00000-of-01024.json.gz"
# Optional Hugging Face DATASET cache folder; None uses its normal default.
HF_DATASET_CACHE = None

# 5. OUTPUT: your results folder. MUST FILL IN. Subfolders are created for you.
output_dir = f"results/{args.model}/{args.dataset}/test"
result_path = f"{output_dir}/ATRMD.npy"

# END OF PATHS. These settings match your partner's learned layer combination.
SEED = 1
TRAIN_SIZE = 5000
BACKGROUND_SIZE = 1000
LABEL_THRESHOLD = 0.3
MAX_NEW_TOKENS = 100
# run.sh's exact layer lists; -1 denotes the final layer.
MODEL_LAYERS = {
    "Llama-3.2-3B-Instruct": list(range(27)) + [-1],
    "Llama-3-8B-Instruct": list(range(31)) + [-1],
    "Gemma-3-4B-it": list(range(34)) + [-1],
}


def prompt_for(dataset, model):
    if dataset in ["nq", "truthfulQA"]:
        instruction = "Answer the following question using exactly one sentence."
    elif dataset == "gsm8k" and model == "Llama-3-8B-Instruct":
        instruction = "Answer the following question using exactly a number. Do not explain."
    elif dataset == "common" and model == "Gemma-3-4B-it":
        instruction = ("Answer the following question by outputting exactly one of: "
                       "A, B, C, D, or E. Do not explain.")
    else:
        instruction = "Answer the following question using exactly one word. Do not explain."
    return instruction + "\nQuestion: {text}\nAnswer:"


class BatchDataset:
    """The same batch-size-one interface used by your partner's UEManager."""
    def __init__(self, questions, answers, token_limits):
        self.x, self.y, self.max_new_tokens = questions, answers, token_limits
        self.batch_size = 1
        if not (len(questions) == len(answers) == len(token_limits)):
            raise ValueError("Questions, answers, and token limits have different lengths.")

    def __len__(self):
        return len(self.x)

    def __iter__(self):
        for i in range(len(self.x)):
            yield self.x[i:i + 1], self.y[i:i + 1], self.max_new_tokens[i:i + 1]

    def select(self, indices):
        return BatchDataset([self.x[int(i)] for i in indices],
                            [self.y[int(i)] for i in indices],
                            [self.max_new_tokens[int(i)] for i in indices])


def read_csv_dataset(path, prompt):
    import pandas as pd

    # Match the original loader's type inference and missing-value handling.
    data = pd.read_csv(path)
    if data.empty:
        raise ValueError(f"Empty CSV: {path}")
    questions = [prompt.format(text=text) for text in data["question"].tolist()]
    answers = [str(answer) for answer in data["correct_answer"].tolist()]
    return BatchDataset(questions, answers, [MAX_NEW_TOKENS] * len(data))


def sample_indices(length, size):
    import numpy as np
    # Match the original dataset sampler, including its NumPy random state.
    np.random.seed(SEED)
    if size == -1 or length < size:
        return np.arange(length)
    return np.random.choice(length, size, replace=False)


def load_background():
    from datasets import load_dataset

    if C4_LOCAL_FILE is not None:
        data = load_dataset("json", data_files=C4_LOCAL_FILE,
                            split="train", cache_dir=HF_DATASET_CACHE)
    else:
        data = load_dataset(C4_DATASET_ID, data_files=C4_SHARD,
                            split="train", cache_dir=HF_DATASET_CACHE)
    # Match utils/dataset.py: first 100,000 records, then <=1,024 characters,
    # then sample up to 1,000 with the same seed and sampling order.
    data = data.select(range(min(100000, len(data))))
    rows = [row for row in data if len(row["text"]) <= 1024]
    if not rows:
        raise ValueError("No C4 records remain after the original length filter.")
    selected = sample_indices(len(rows), BACKGROUND_SIZE)
    return BatchDataset([rows[int(i)]["text"] for i in selected],
                        [rows[int(i)]["url"] for i in selected],
                        [MAX_NEW_TOKENS] * len(selected))


def make_atrmd_estimators(layers, training_labels):
    """Inline the two original distance classes, retaining only ATRMD behavior."""
    import numpy as np
    import torch
    from lm_polygraph.estimators.estimator import Estimator
    from lm_polygraph.estimators.mahalanobis_distance import (
        compute_inv_covariance,
        create_cuda_tensor_from_numpy,
        mahalanobis_distance_with_known_centroids_sigma_inv,
    )

    class ATRMD(Estimator):
        def __init__(self, layer):
            self.hidden_layer = layer
            self.embeddings_type = "decoder"
            self.aggregation = "mean"
            self.suffix = "" if layer == -1 else f"_{layer}"
            super().__init__([
                f"token_embeddings{self.suffix}",
                f"train_token_embeddings{self.suffix}",
                f"background_train_token_embeddings{self.suffix}",
                "greedy_tokens", "train_greedy_tokens", "train_target_texts",
            ], "sequence")
            self.is_fitted = False

        def __str__(self):
            return f"ATRMD_decoder{self.suffix}"

        def __call__(self, stats):
            if not self.is_fitted:
                train = create_cuda_tensor_from_numpy(
                    stats[f"train_token_embeddings_decoder{self.suffix}"])
                background = create_cuda_tensor_from_numpy(
                    stats[f"background_train_token_embeddings_decoder{self.suffix}"])
                train_lengths = [len(tokens) for tokens in stats["train_greedy_tokens"]]
                if len(train_lengths) != len(training_labels):
                    raise ValueError("Training labels do not match generated training examples.")
                token_labels = np.repeat(training_labels, train_lengths)
                if len(token_labels) != len(train):
                    raise ValueError("Training token labels do not match token embeddings.")
                # Preserve the original >10-token fallback exactly.
                keep = token_labels >= LABEL_THRESHOLD
                if LABEL_THRESHOLD > 0 and keep.sum() > 10:
                    train = train[torch.as_tensor(keep, device=train.device)]
                self.train_mean = train.mean(axis=0)
                self.train_inverse, _ = compute_inv_covariance(
                    self.train_mean.unsqueeze(0), train)
                self.background_mean = background.mean(axis=0)
                self.background_inverse, _ = compute_inv_covariance(
                    self.background_mean.unsqueeze(0), background)
                self.is_fitted = True

            embeddings = create_cuda_tensor_from_numpy(
                stats[f"token_embeddings_decoder{self.suffix}"])
            distances = []
            for mean, inverse in [(self.train_mean, self.train_inverse),
                                  (self.background_mean, self.background_inverse)]:
                distances.append(mahalanobis_distance_with_known_centroids_sigma_inv(
                    mean.float(), None, inverse.float(), embeddings.float()
                )[:, 0].cpu().detach().numpy())
            relative = distances[0] - distances[1]
            token_counts = [len(tokens) for tokens in stats["greedy_tokens"]]
            if sum(token_counts) != len(relative):
                raise ValueError("Generated tokens and token embeddings do not align.")
            scores, absolute_scores, offset = [], [], 0
            for count in token_counts:
                scores.append(relative[offset:offset + count].mean())
                absolute_scores.append(distances[0][offset:offset + count].mean())
                offset += count
            self.absolute_scores = np.asarray(absolute_scores)
            return np.asarray(scores)

    return [ATRMD(layer) for layer in layers]


def make_atrmd_aggregated_estimator(layers, training_labels):
    """The RelativeTokenMahalanobis + LinReg branch enabled by run.sh."""
    import numpy as np
    from sklearn.decomposition import PCA
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import train_test_split
    from lm_polygraph.estimators.estimator import Estimator

    class ATRMD_aggregated(Estimator):
        def __init__(self):
            self.hidden_layers = layers
            dependencies = ["greedy_tokens", "train_greedy_tokens", "train_target_texts"]
            for layer in layers:
                suffix = "" if layer == -1 else f"_{layer}"
                dependencies += [f"token_embeddings{suffix}",
                                 f"train_token_embeddings{suffix}",
                                 f"background_train_token_embeddings{suffix}"]
            super().__init__(dependencies, "sequence")
            self.is_fitted = False

        def __str__(self):
            return "ATRMD_aggregated"

        def __call__(self, stats):
            if not self.is_fitted:
                tokens = stats["train_greedy_tokens"]
                if len(tokens) != len(training_labels):
                    raise ValueError("Training labels do not match generated training examples.")
                # Same internal train/dev split as LinRegTokenMahalanobisDistance.
                train_idx, dev_idx = train_test_split(
                    list(range(len(tokens))), test_size=0.5, shuffle=True, random_state=42)
                offsets = np.cumsum([0] + [len(row) for row in tokens])
                train_token_idx = np.concatenate([
                    np.arange(offsets[i], offsets[i + 1]) for i in train_idx])
                dev_token_idx = np.concatenate([
                    np.arange(offsets[i], offsets[i + 1]) for i in dev_idx])

                # The original runner caches full-training token labels in its
                # per-layer methods, then slices that cache by train_token_idx.
                # Passing the corresponding example labels gives the same mask
                # without running or saving the separate per-layer baselines.
                self.layer_estimators = make_atrmd_estimators(
                    layers, training_labels[train_idx])
                train_mds, absolute_train_mds = [], []
                for layer, estimator in zip(layers, self.layer_estimators):
                    suffix = "" if layer == -1 else f"_{layer}"
                    embeddings = stats[f"train_token_embeddings_decoder{suffix}"]
                    split_stats = {
                        "train_greedy_tokens": [tokens[i] for i in train_idx],
                        "greedy_tokens": [tokens[i] for i in dev_idx],
                        f"train_token_embeddings_decoder{suffix}":
                            [embeddings[i] for i in train_token_idx],
                        f"token_embeddings_decoder{suffix}":
                            [embeddings[i] for i in dev_token_idx],
                        f"background_train_token_embeddings_decoder{suffix}":
                            stats[f"background_train_token_embeddings_decoder{suffix}"],
                    }
                    train_mds.append(estimator(split_stats))
                    absolute_train_mds.append(estimator.absolute_scores)

                train_dists = np.asarray(train_mds).T
                train_dists[np.isnan(train_dists)] = 0
                # The runner fits the plain-distance combination first. Its PCA
                # can consume NumPy randomness (the auto solver is randomized
                # for some data sizes). Preserve that step before the relative
                # PCA so removing the other output does not change these scores.
                absolute_train_dists = np.asarray(absolute_train_mds).T
                absolute_train_dists[np.isnan(absolute_train_dists)] = 0
                PCA(n_components=10).fit_transform(absolute_train_dists)
                # run.sh selects norm="orig", remove_corr=True, remove_alg=3,
                # sim_pca=False, n_components=10, meta_model="LinReg".
                self.pca = PCA(n_components=10)
                features = self.pca.fit_transform(train_dists)
                target = np.asarray(training_labels[dev_idx], dtype=np.float64).copy()
                target[np.isnan(target)] = 0
                self.regressor = Ridge(positive=False)
                self.regressor.fit(features, 1 - target)
                self.is_fitted = True

            # The original inner MD objects stay fitted to the training half
            # after fitting the regressor. Keep those same references for test.
            eval_dists = np.asarray([estimator(stats)
                                     for estimator in self.layer_estimators]).T
            eval_dists[np.isnan(eval_dists)] = 0
            return self.regressor.predict(self.pca.transform(eval_dists))

    return ATRMD_aggregated()


def collect_scores(test_data, estimator, estimations):
    import numpy as np

    total = len(test_data)
    key = ("sequence", str(estimator))
    values = np.asarray(estimations[key])
    if values.shape != (total,) or not np.isfinite(values).all():
        raise ValueError(f"Invalid or incomplete output for {key}: shape={values.shape}")
    return values


if __name__ == "__main__":
    # Print the configured paths before loading the model.
    input_files = [train_data_path, test_data_path, labels_path]
    if C4_LOCAL_FILE is not None:
        input_files.append(C4_LOCAL_FILE)
    for path in input_files:
        print(f"INPUT: {path}")
    print(f"MODEL: {model_path}")
    print(f"OUTPUT: {result_path}")

    # run.sh makes only the selected GPU visible and loads with device_map="auto".
    gpu_id = "0" if args.device == "cuda" else args.device.removeprefix("cuda:")
    if not gpu_id.isdigit():
        raise SystemExit("Use --device cuda:0, cuda:1, etc.")
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

    # Heavy imports happen after selecting the GPU.
    import numpy as np
    import torch
    from transformers import set_seed
    from lm_polygraph.utils.manager import UEManager
    from lm_polygraph.utils.model import WhiteboxModel
    from lm_polygraph.utils.processor import Logger

    logging.basicConfig(level=logging.INFO)
    if not torch.cuda.is_available():
        raise SystemExit("This baseline requires a CUDA GPU; select it with --device cuda:0.")
    torch.cuda.set_device(0)
    set_seed(SEED)

    # Match the original order: initialize the model before loading/sampling data.
    model = WhiteboxModel.from_pretrained(
        model_path, {"generate_until": ["\n"], "samples_n": 5},
        device_map="auto", add_bos_token=True, attn_implementation="eager")

    prompt = prompt_for(args.dataset, args.model)
    test_data = read_csv_dataset(test_data_path, prompt)
    train_data = read_csv_dataset(train_data_path, prompt)
    labels_payload = torch.load(labels_path, map_location="cpu", weights_only=False)
    labels = np.asarray(labels_payload["y_train"], dtype=np.float64).reshape(-1)
    if len(labels) != len(train_data):
        raise ValueError("y_train must contain one label per training CSV row.")
    train_indices = sample_indices(len(train_data), TRAIN_SIZE)
    train = train_data.select(train_indices)
    labels = labels[train_indices]
    background = load_background()

    layers = MODEL_LAYERS[args.model]
    print(f"Training examples: {len(train)}; background examples: {len(background)}")
    print(f"Test examples: {len(test_data)}; layers: {layers}")
    print("ATRMD.npy will contain one combined score per test example.")

    estimator = make_atrmd_aggregated_estimator(layers, labels)
    manager = UEManager(
        test_data, model, [estimator], [], [], [Logger()],
        train_data=train, background_train_data=background, ignore_exceptions=False,
        deberta_batch_size=1 if args.dataset == "gsm8k" else 10, ensemble_model=None)
    manager()
    scores = collect_scores(test_data, estimator, manager.estimations)

    # Save one aggregated ATRMD score per test example, in the test CSV's original order.
    os.makedirs(output_dir, exist_ok=True)
    np.save(result_path, scores)
    print(f"Saved {result_path}: {scores.shape}")
