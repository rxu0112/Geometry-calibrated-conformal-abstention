# Reproduction guide

This guide covers the full experiment configurations, script inputs and outputs, and ablations. See the [README](../readme.md) for the method overview and main experiment commands, and its [environment section](../readme.md#environment) for installation.

- [Models and data](#running-the-code)
- [Main experiment](#main-experiment)
- [Ablation experiments](#ablation-experiments)
- [Output and rerun conventions](#output-and-rerun-conventions)

## Running the code

Run every command from the repository root, using `python -m` and the module names shown below. This makes the local imports available and resolves the relative `processed_data/` and `results/` paths correctly.

```bash
cd Geometry-calibrated-conformal-abstention
```

Download the complete `results/` folder from the [Hugging Face bucket](https://huggingface.co/buckets/rxu0112/Geometry-calibrated-conformal-abstention) and place it at the repository root, alongside `processed_data/`, preserving its directory structure.

Before recomputing saved outputs, follow the [rerun conventions](#output-and-rerun-conventions).

Extraction, correctness evaluation, and baseline scoring scripts accept command-line arguments and run one model/dataset pair at a time. Feature computation, calibrator training, baseline data preparation, and conformal evaluation loop over all six datasets and all three models by default. Most ablations also use settings inside their Python files; their scopes are described below. GPU settings default to `cuda:0`; scripts with a `--device` argument allow overriding it.

### Models and datasets

Use the following model identifiers in `--model` arguments, filenames, and result directories:

| Model identifier | Hugging Face checkpoint |
|---|---|
| `Llama-3.2-3B-Instruct` | `meta-llama/Llama-3.2-3B-Instruct` |
| `Llama-3-8B-Instruct` | `meta-llama/Meta-Llama-3-8B-Instruct` |
| `Gemma-3-4B-it` | `google/gemma-3-4b-it` |

Dataset identifiers are `sciq`, `simple_questions_wiki`, `nq`, `truthfulQA`, `gsm8k`, and `common`. Use these exact spellings throughout the workflow.

Each dataset's input is `processed_data/<dataset>/prepared.json`. Despite the `.json` extension, the provided files use JSON Lines: one object per line, with `question` and `correct_answer` fields.

### Repository layout

```text
Geo-CA/                 Feature computation, confidence calibration, and conformal evaluation
llama_representation/   Llama logit and contribution extraction
gemma_representation/   Gemma logit and contribution extraction
correctness_eval/      HHEM and GPT correctness evaluation
baselines/             Baseline data preparation and scoring
ablation/              Ablation experiments and comparison tables
processed_data/        Prepared questions, CSV splits, labels, and split indices
results/               Extraction outputs, features, judgments, scores, and evaluations
general_utils.py       Shared mathematical and pickle utilities
```

Most outputs follow this structure:

```text
processed_data/<dataset>/
    prepared.json
    <model>_<split>_data.csv
    <model>_labels.pt
    <model>_indices.pt

results/<model>/<dataset>/
    <dataset>_lens_results.pkl
    <dataset>_contri_results_manhattan.pkl
    <dataset>_hhem_judge.pt
    <dataset>_gpt-4o_judge.json
    <dataset>_gpt-5.1_judge.json
    <dataset>_features.pt
    <dataset>_pca_features.pt
    train/
    val/
    test/
```

Judgment files depend on the dataset, as described below. CSV splits include `ref`, `train`, `val`, and `test`.

## Main experiment

The main experiment covers all six datasets and all three models. Feature computation, calibrator training, baseline data preparation, and conformal evaluation loop over all 18 configurations by default.

Extraction, correctness evaluation, and baseline scoring accept one model/dataset pair per run. For example, the commands below use `sciq` and `Llama-3.2-3B-Instruct`. The extraction module selects the model; correctness evaluators and baseline scorers select it with `--model`.

Run extraction, correctness evaluation, and baseline scoring separately for each required model/dataset combination. Complete each stage for all required configurations before running a script that processes them together.

### 1. Extract logits, hidden states, and contributions

Use the extraction modules for your model:

| Model | Logit module | Contribution module |
|---|---|---|
| `Llama-3.2-3B-Instruct` | `llama_representation.llama_3B_logit` | `llama_representation.llama_3B_contri` |
| `Llama-3-8B-Instruct` | `llama_representation.llama_8B_logit` | `llama_representation.llama_8B_contri` |
| `Gemma-3-4B-it` | `gemma_representation.gemma_4B_logit` | `gemma_representation.gemma_4B_contri` |

For example, extract logits and hidden states for SciQ with `Llama-3.2-3B-Instruct`:

```bash
conda activate Geo_CA_env
python -m llama_representation.llama_3B_logit --dataset sciq --device cuda:0
```

For the same model and dataset, extract attention and MLP contributions:

```bash
conda activate Geo_CA_env
python -m llama_representation.llama_3B_contri --dataset sciq --metric manhattan --device cuda:0
```

Both scripts read `processed_data/<dataset>/prepared.json`. The logit script saves probabilities, logits, hidden states, and generated tokens to `<dataset>_lens_results.pkl`. The contribution script saves attention/MLP contributions and predictions to `<dataset>_contri_results_<metric>.pkl`. Both files are stored under `results/<model>/<dataset>/`.

Use `--metric manhattan` for the main workflow: the correctness and baseline scripts read the Manhattan contribution file. Contribution extraction also supports `dot`; feature computation's `metric` setting must match the chosen file. The contribution scripts use `Qwen/Qwen2.5-7B` to form factual statements for short-answer evaluation.

Llama extraction supports `--dtype fp16`, `bf16`, or `fp32`; Gemma supports `bf16` or `fp32`. Contribution scripts also accept `--i_block` to control computation block size. NQ extraction uses the first 5,000 prepared examples.

### 2. Evaluate answer correctness

Choose the evaluation route for the dataset:

| Datasets | Evaluation route |
|---|---|
| `sciq`, `simple_questions_wiki`, `gsm8k` | Run `correctness_eval.short_hhem`, then `correctness_eval.short_gpt` with each judge model. |
| `common` | Run `correctness_eval.short_hhem`. |
| `nq`, `truthfulQA` | Run `correctness_eval.long_gpt` with each judge model. |

For example, evaluate SciQ answers from `Llama-3.2-3B-Instruct`:

```bash
python -m correctness_eval.short_hhem --dataset sciq --model Llama-3.2-3B-Instruct
python -m correctness_eval.short_gpt --dataset sciq --model Llama-3.2-3B-Instruct
```

Before running either GPT script, configure its API client settings. Set `judgement_model` to `gpt-4o` and run it, then set it to `gpt-5.1` and run it again. Run the `short_gpt` example above twice, once with each setting. Feature computation requires both judge files for every dataset except `common`.

For other model/dataset pairs, use `python -m` with the appropriate module from the table and supply `--dataset` and `--model`. Use `correctness_eval.long_gpt` for `nq` and `truthfulQA`.

HHEM reads the Manhattan contribution file and writes `<dataset>_hhem_judge.pt`. The short-answer GPT script evaluates examples whose HHEM score is below `0.2`; the long-answer GPT script evaluates the extracted answers directly. Both GPT scripts also read the prepared questions and save `<dataset>_<judgement_model>_judge.json` under the model/dataset result directory.

### 3. Compute Geo-CA features

`Geo-CA/feature_computation.py` computes features for all 18 model/dataset combinations. Keep `metric = 'manhattan'` to match the contribution extraction commands, then run:

```bash
python -m Geo-CA.feature_computation
```

This combines extraction outputs with the required judgments, filters examples, creates the reference/remain split, and computes contribution, angular-change, and hidden-state geometry features. It writes:

```text
results/<model>/<dataset>/<dataset>_features.pt
```

Each pair requires its own matching judgments and extraction outputs before running this command.

### 4. Train the confidence calibrator

`Geo-CA/calibrator_training.py` trains a calibrator for each of the 18 configurations:

```bash
python -m Geo-CA.calibrator_training
```

The script reads `<dataset>_features.pt`, trains an XGBoost calibrator using geometric distances and token confidence, and saves the test labels and answer-level confidence scores:

```text
results/<model>/<dataset>/test/label.npy
results/<model>/<dataset>/test/proposed.npy
```

The remaining examples after the reference split are divided into train/validation/test subsets in a 60/20/20 ratio using stratified splits with `random_state=42`.

### 5. Prepare and score the baselines

Use `baselines/baseline_data_prepare.py` to prepare all 18 configurations:

```bash
conda activate baseline_env
python -m baselines.baseline_data_prepare
```

This reads `prepared.json` and the Geo-CA feature file to create matching CSV splits, labels, and split indices in `processed_data/<dataset>/` for each pair.

For example, score SciQ with `Llama-3.2-3B-Instruct`:

```bash
python -m baselines.AttentionScore --dataset sciq --model Llama-3.2-3B-Instruct --device cuda:0
python -m baselines.Perplexity --dataset sciq --model Llama-3.2-3B-Instruct --device cuda:0
python -m baselines.SemanticEntropy --dataset sciq --model Llama-3.2-3B-Instruct --device cuda:0
python -m baselines.SAR --dataset sciq --model Llama-3.2-3B-Instruct --device cuda:0
python -m baselines.EigenScore --dataset sciq --model Llama-3.2-3B-Instruct --device cuda:0
python -m baselines.Focus --dataset sciq --model Llama-3.2-3B-Instruct --device cuda:0
python -m baselines.PTrue --dataset sciq --model Llama-3.2-3B-Instruct --device cuda:0
python -m baselines.ATRMD --dataset sciq --model Llama-3.2-3B-Instruct --device cuda:0
```

The first seven scripts score `train`, `val`, and `test`, using the split indices and Manhattan contribution output to align the generations. Each writes `results/<model>/<dataset>/<split>/<baseline>.npy`. For full reproduction, repeat these scoring commands with each of the 18 model/dataset combinations.

ATRMD learns a layer combination from the training data and scores only the test split, writing `test/ATRMD.npy`. It also uses C4 background data; its `C4_LOCAL_FILE` setting allows selecting a local shard. Focus builds an IDF cache under `idf_cache/` using its configured reference dataset.

### 6. Evaluate conformal abstention

Set `calibrated_baseline = False` in `Geo-CA/conformal_abstaining.py` and run the evaluation for all 18 configurations:

```bash
conda activate Geo_CA_env
python -m Geo-CA.conformal_abstaining
```

The script reads test labels, Geo-CA scores, and baseline scores, and evaluates repeated calibration/test splits. Results include AUROC, AUPRC, retention/conditional-correctness statistics, and conformal bounds. They are saved to:

```text
results/<model>/<dataset>/test/comparison.pkl
```

This mode compares Geo-CA with AttentionScore, Perplexity, SemanticEntropy, SAR, EigenScore, Focus, P(True), and ATRMD. The script converts uncertainty scores to confidence scores internally and filters nonfinite scores before evaluation. With the default loops, it evaluates all 18 model/dataset combinations.

## Ablation experiments

Run all ablations in `Geo_CA_env`:

```bash
conda activate Geo_CA_env
```

The calibrated-baseline, PCA, distance, permutation, and balanced experiments cover all six datasets and all three models by default. The corresponding comparison-table commands follow the experiment commands below and cover the same configurations.

Generalization trains and evaluates across datasets. Domain shift, calibration-set size, and WRCP use the specific experimental configurations described in their sections; those selections are intentional.

### Calibrated baselines

`ablation/baseline_calibration_ablation.py` calibrates the seven baselines across all 18 configurations:

```bash
python -m ablation.baseline_calibration_ablation
```

This reads the baseline train/validation/test scores and prepared labels, then writes `results/<model>/<dataset>/test/<baseline>_calibrated.npy` for AttentionScore, Perplexity, SemanticEntropy, SAR, EigenScore, Focus, and PTrue.

Set `calibrated_baseline = True` in `Geo-CA/conformal_abstaining.py` and evaluate all 18 configurations:

```bash
python -m Geo-CA.conformal_abstaining
```

The output is `test/comparison_calibrated_baseline.pkl`. This run compares Geo-CA with the seven calibrated baselines. ATRMD is excluded because it already requires training and is not a training-free baseline.

### PCA and confidence-only calibration

`ablation/pca_feature_computation.py` computes PCA features for all 18 model/dataset combinations by default:

```bash
python -m ablation.pca_feature_computation
```

This reads the logit/hidden-state output and dataset-appropriate judgments, and writes `results/<model>/<dataset>/<dataset>_pca_features.pt`.

The `CONF_ONLY` setting in `ablation/calibrator_pca_ablation.py` selects the experiment:

| Setting | Test outputs |
|---|---|
| `CONF_ONLY = True` | `label_conf_only.npy`, `proposed_conf_only.npy` |
| `CONF_ONLY = False` | `label_pca.npy`, `proposed_pca.npy` |

Set `CONF_ONLY = False` for the PCA experiment, then run the calibrator and its comparison table:

```bash
python -m ablation.calibrator_pca_ablation
python -m ablation.table_pca_ablation
```

For confidence-only calibration, set `CONF_ONLY = True` and rerun the calibrator. It reads PCA feature files in both modes. To print the confidence-only comparison table, replace the `label_pca.npy` and `proposed_pca.npy` inputs in `ablation/table_pca_ablation.py` with `label_conf_only.npy` and `proposed_conf_only.npy`, then rerun the table command.

### Distance functions

```bash
python -m ablation.calibrator_distance_ablation
python -m ablation.table_distance_ablation
```

The calibrator reads the main feature files and tests `knn`, `shrinkage_mahalanobis`, `diagonal_mahalanobis`, and `standardized_euclidean`, implemented in `ablation/distance_functions.py`. It writes `test/proposed_<selected_distance>.npy` and the matching `test/label.npy`.

The calibrator and table script both cover all six datasets and all three models by default.

### Feature permutation and balanced feature groups

```bash
python -m ablation.calibrator_permutation_ablation
python -m ablation.calibrator_balanced_ablation
python -m ablation.table_balanced_ablation
```

Both calibrators read the main feature files. The permutation experiment perturbs feature groups during inference and saves AUROC/AUPRC summaries in `test/proposed_permutation.json`. The balanced experiment normalizes distances within feature groups and writes `test/label_balanced.npy` and `test/proposed_balanced.npy`. The table script compares the balanced variant with the main method.

### Generalization and domain shift

```bash
python -m ablation.calibrator_generalization_ablation
python -m ablation.conformal_abstaining_domain_shift
```

The first script trains on each of the six datasets and evaluates on all six datasets for each of the three models, using the main feature files. It stores predictions, labels, and AUROC/AUPRC results in `results/calibrator_generalization_ablation.pkl`.

The domain-shift experiment intentionally uses `simple_questions_wiki` as the source, the other five datasets as targets, and all three models. Its script reads the generalization result file and prints conformal-abstention results with `TARGET_PARTICIPATION = 0.5`. Keep these settings to reproduce this experiment.

### Calibration-set size

```bash
python -m ablation.conformal_abstaining_cal_size_ablation
```

This experiment intentionally uses `Llama-3.2-3B-Instruct` on `simple_questions_wiki`, set by the scalar `dataset` and `model` variables inside `main()`. It reads that pair's `test/label.npy` and `test/proposed.npy` and prints results across `CALIBRATION_RATIOS`. `PARTICIPATION_LEVELS` and `N_TRIALS` control the evaluation. Keep the selected pair to reproduce this experiment.

### WRCP calibrator

```bash
python -m ablation.calibrator_WRCP_ablation
```

This experiment intentionally uses `Gemma-3-4B-it` on `simple_questions_wiki` and `nq` as two domains. The script reads their main feature files, runs a Wasserstein-regularized MLP calibrator, and prints conformal results. Keep this model and domain pair to reproduce the experiment. It does not save a separate result file.

## Output and rerun conventions

- `label.npy` and each associated score array must refer to the same test examples in the same order. Generate baseline CSVs and indices from the corresponding feature file.
- Use the model identifier from the model table for data/result paths. The Hugging Face repository ID is used for loading model weights and tokenizers.
- Logit, contribution, and GPT evaluation scripts resume from existing outputs. Conformal evaluation skips methods already present in its comparison file. To recompute after changing inputs or settings, move the previous outputs aside before rerunning the affected stage.
- Some supplied main comparison files also contain a redundant historical `ATMD` entry. The current baseline script produces `ATRMD`, and the current evaluator selects `ATRMD`. Select the desired method names when reading cached comparisons.
- The table scripts print their comparisons to the console. The calibration-size, domain-shift, and WRCP scripts also print their results rather than saving new evaluation files.
