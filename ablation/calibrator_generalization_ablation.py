import json
import matplotlib.pyplot as plt
import torch
import numpy as np
from general_utils import *
from pandas.plotting import parallel_coordinates
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, roc_auc_score, accuracy_score, roc_curve, auc, \
    precision_recall_curve, log_loss
import xgboost as xgb
import optuna
from tqdm import tqdm

def feature_label_extraction(model,dataset):
    statistic_results = torch.load(f"results/{model}/{dataset}/{dataset}_features.pt", weights_only=False)
    hem_examples = statistic_results['hem_examples']
    contri_examples = statistic_results['contri_examples']
    angle_diff_examples = statistic_results['angle_diff_examples']
    prob_examples = statistic_results['prob_examples']
    cos_to_right_hidden_avg_examples = statistic_results['cos_to_right_hidden_avg_examples']
    cos_to_wrong_hidden_avg_examples = statistic_results['cos_to_wrong_hidden_avg_examples']
    ref_idx = statistic_results['ref_idx']
    remain_idx = statistic_results['remain_idx']

    right_threshold = 0.8
    wrong_threshold = 0.2
    y = np.array([1 if s > right_threshold else 0 for s in hem_examples])
    remain_y = y[remain_idx]
    remain_prob_examples = [prob_examples[r] for r in remain_idx]


    right_ref_contri_examples = [contri_examples[r] for r in ref_idx if hem_examples[r] > right_threshold]
    wrong_ref_contri_examples = [contri_examples[r] for r in ref_idx if hem_examples[r] < wrong_threshold]
    right_ref_angle_diff_examples = [angle_diff_examples[r] for r in ref_idx if hem_examples[r] > right_threshold]
    wrong_ref_angle_diff_examples = [angle_diff_examples[r] for r in ref_idx if hem_examples[r] < wrong_threshold]
    right_ref_cos_to_right_hidden_avg_examples = [cos_to_right_hidden_avg_examples[r] for r in ref_idx if hem_examples[r] > right_threshold]
    wrong_ref_cos_to_right_hidden_avg_examples = [cos_to_right_hidden_avg_examples[r] for r in ref_idx if hem_examples[r] < wrong_threshold]
    right_ref_cos_to_wrong_hidden_avg_examples = [cos_to_wrong_hidden_avg_examples[r] for r in ref_idx if hem_examples[r] > right_threshold]
    wrong_ref_cos_to_wrong_hidden_avg_examples = [cos_to_wrong_hidden_avg_examples[r] for r in ref_idx if hem_examples[r] < wrong_threshold]

    right_ref_metric_distribution = np.concatenate([np.column_stack(right_ref_contri_examples).T,
                                                    np.column_stack(right_ref_angle_diff_examples).T,
                                                    np.column_stack(right_ref_cos_to_right_hidden_avg_examples).T,
                                                    np.column_stack(right_ref_cos_to_wrong_hidden_avg_examples).T,
                                                    ],
                                                axis=1)

    wrong_ref_metric_distribution = np.concatenate([np.column_stack(wrong_ref_contri_examples).T,
                                                    np.column_stack(wrong_ref_angle_diff_examples).T,
                                                    np.column_stack(wrong_ref_cos_to_right_hidden_avg_examples).T,
                                                    np.column_stack(wrong_ref_cos_to_wrong_hidden_avg_examples).T,
                                                    ],
                                                axis=1)

    right_mu, right_Sigma_inv = precompute_mahalanobis(right_ref_metric_distribution)
    wrong_mu, wrong_Sigma_inv = precompute_mahalanobis(wrong_ref_metric_distribution)

    remain_contri_examples = [contri_examples[r] for r in remain_idx]
    remain_angle_diff_examples = [angle_diff_examples[r] for r in remain_idx]
    remain_cos_to_right_hidden_avg_examples = [cos_to_right_hidden_avg_examples[r] for r in remain_idx]
    remain_cos_to_wrong_hidden_avg_examples = [cos_to_wrong_hidden_avg_examples[r] for r in remain_idx]

    coordinates = []
    for s_idx in tqdm(range(len(remain_idx))):
        distance_to_right = []
        distance_to_wrong = []
        for t_idx in range(len(remain_prob_examples[s_idx])):
            metric = np.concatenate([remain_contri_examples[s_idx][:, t_idx],
                                    remain_angle_diff_examples[s_idx][:, t_idx],
                                    remain_cos_to_right_hidden_avg_examples[s_idx][:, t_idx],
                                    remain_cos_to_wrong_hidden_avg_examples[s_idx][:, t_idx],
                                    ])
            distance_to_right.append(mahalanobis_distance(metric, right_mu, right_Sigma_inv))
            distance_to_wrong.append(mahalanobis_distance(metric, wrong_mu, wrong_Sigma_inv))
        coords = np.stack([distance_to_right, distance_to_wrong], axis=1)
        coordinates.append(coords)

    # Calibrator Training
    features = []
    for coords, prob in zip(coordinates, remain_prob_examples):
        prob = prob.reshape(-1, 1)   # (n, 1)
        features.append(np.concatenate([coords, prob], axis=1))

    indices = np.arange(len(remain_y))

    temp_idx, test_idx, y_temp, y_test = train_test_split(
        indices, remain_y, test_size=0.2, random_state=42, stratify=remain_y)

    train_idx, val_idx, y_train, y_val = train_test_split(
        temp_idx, y_temp, test_size=0.25, random_state=42, stratify=y_temp)

    train_features = [features[i] for i in train_idx]
    test_features = [features[i] for i in test_idx]
    val_features = [features[i] for i in val_idx]

    X = np.concatenate(features,axis=0)
    y_broadcast = np.concatenate([
        np.full(len(features[i]), remain_y[i])
        for i in indices
    ])

    X_train = np.concatenate(train_features, axis=0)
    y_train_broadcast = np.concatenate([
        np.full(len(features[i]), remain_y[i])
        for i in train_idx
    ])

    X_val = np.concatenate(val_features, axis=0)
    y_val_broadcast = np.concatenate([
        np.full(len(features[i]), remain_y[i])
        for i in val_idx
    ])

    X_test = np.concatenate(test_features, axis=0)
    y_test_broadcast = np.concatenate([
        np.full(len(features[i]), remain_y[i])
        for i in test_idx
    ])

    scaler = StandardScaler()
    scaler.fit(X_train)

    X_train = scaler.transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)
    return {
        "X_train": X_train.astype(np.float32),
        "X_val": X_val.astype(np.float32),
        "X_test": X_test.astype(np.float32),

        # Response-level labels
        "y_train": np.asarray(y_train),
        "y_val": np.asarray(y_val),
        "y_test": np.asarray(y_test),

        # Token-level labels aligned with X_*
        "y_train_broadcast": y_train_broadcast,
        "y_val_broadcast": y_val_broadcast,
        "y_test_broadcast": y_test_broadcast,

        "train_features": train_features,
        "test_features": test_features,}

device='cuda:0'
feature_label_data = {}
for dataset in ['sciq', 'nq', 'common', 'simple_questions_wiki', 'truthfulQA', 'gsm8k']: # 'sciq', 'nq', 'common', 'simple_questions_wiki', 'truthfulQA', 'gsm8k'
    for model in ['Llama-3.2-3B-Instruct', 'Gemma-3-4B-it', 'Llama-3-8B-Instruct']: # Llama-3.2-3B-Instruct, Gemma-3-4B-it, Llama-3-8B-Instruct
        feature_label_data.setdefault(model, {})[dataset] = (feature_label_extraction(model, dataset))

from pathlib import Path
result_path = Path("results/calibrator_generalization_ablation.pkl")
cross_domain_results = {}

for train_set in ['sciq', 'nq', 'common', 'simple_questions_wiki', 'truthfulQA', 'gsm8k']: # 'sciq', 'nq', 'common', 'simple_questions_wiki', 'truthfulQA', 'gsm8k'
    for model in ['Llama-3.2-3B-Instruct', 'Gemma-3-4B-it', 'Llama-3-8B-Instruct']: 
        X_train = feature_label_data[model][train_set]['X_train']
        X_val = feature_label_data[model][train_set]['X_val']
        y_train = feature_label_data[model][train_set]['y_train']
        y_val = feature_label_data[model][train_set]['y_val']
        y_train_broadcast = feature_label_data[model][train_set]['y_train_broadcast']
        y_val_broadcast = feature_label_data[model][train_set]['y_val_broadcast']
        train_features = feature_label_data[model][train_set]['y_val_broadcast']

        dtrain = xgb.DMatrix(X_train, label=y_train_broadcast)
        dval = xgb.DMatrix(X_val, label=y_val_broadcast)
        
        for metric in ["auc"]:
            def objective(trial, metric = metric):
                params = {
                    "objective": "binary:logistic",
                    "eval_metric": metric,
                    "device": device,
                    "max_depth": trial.suggest_int("max_depth", 2, 6),
                    "eta": trial.suggest_float("eta", 0.01, 0.2, log=True),
                    "subsample": trial.suggest_float("subsample", 0.6, 0.9),
                    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 0.9),
                    "lambda": trial.suggest_float("lambda", 0.1, 10.0, log=True),
                    "alpha": trial.suggest_float("alpha", 0.1, 10.0, log=True),
                    "scale_pos_weight": trial.suggest_float("scale_pos_weight", 0.3, 1.5),
                    "seed": 42
                }

                evals = [(dtrain, "train"), (dval, "val")]

                bst = xgb.train(
                    params=params,
                    dtrain=dtrain,
                    num_boost_round=5000,
                    evals=evals,
                    early_stopping_rounds=50,
                    verbose_eval=False
                )

                best_round = bst.best_iteration  # <-- best round for this trial
                y_proba_val = bst.predict(dval)

                if metric == 'auc':
                    val_auc = roc_auc_score(y_val_broadcast, y_proba_val)
                    val_auprc = average_precision_score(y_val_broadcast, y_proba_val)
                    score = val_auc+val_auprc
                elif metric == 'logloss':
                    score = log_loss(y_val_broadcast, y_proba_val)
                elif metric == 'error':
                    y_pred_binary = (y_proba_val >= 0.5).astype(int)
                    score = (y_pred_binary != y_val_broadcast).mean()
                    
                # You can store best_round in trial user attributes
                trial.set_user_attr("best_round", best_round)

                return score

            if metric == 'auc':
                study = optuna.create_study(direction="maximize")
            else:
                study = optuna.create_study(direction="minimize")

            study.optimize(objective, n_trials=50)

            print("Best validation AUROC:", study.best_value)
            print("Best params:", study.best_params)

            dtrain_full = xgb.DMatrix(np.vstack([X_train, X_val]), label=np.hstack([y_train_broadcast, y_val_broadcast]))
            final_params = {
                            'device':device,
                            **study.best_params,
                        }
            bst = xgb.train(
                params=final_params,
                dtrain=dtrain_full,
                num_boost_round=study.best_trial.user_attrs["best_round"],
                verbose_eval=False
            )

            for test_set in ['sciq', 'nq', 'common', 'simple_questions_wiki', 'truthfulQA', 'gsm8k']:
                X_test = feature_label_data[model][test_set]['X_test']
                y_test = feature_label_data[model][test_set]['y_test']
                y_test_broadcast = feature_label_data[model][test_set]['y_test_broadcast']
                test_features = feature_label_data[model][test_set]['test_features']

                dtest = xgb.DMatrix(X_test, label=y_test_broadcast)

                y_proba_test_broadcast = bst.predict(dtest)
                y_proba_test = []
                start = 0
                for feats in test_features:
                    n = len(feats)
                    seq = y_proba_test_broadcast[start:start + n]
                    y_proba_test.append(np.log(np.clip(seq, 1e-12, 1.0)).mean())
                    start += n

                auprc_test = average_precision_score(y_test, y_proba_test)
                print('test auprc:', auprc_test)
                auroc_test = roc_auc_score(y_test, y_proba_test)
                print('test auroc:', auroc_test)

                record = {
                    "model": model,
                    "train_set": train_set,
                    "test_set": test_set,
                    "auprc": float(auprc_test),
                    "auroc": float(auroc_test),
                    "y_proba_test": y_proba_test,
                    "y_test": y_test.astype(np.int8),
                }

                cross_domain_results.setdefault(
                    model, {}
                ).setdefault(
                    train_set, {}
                )[test_set] = record

                with result_path.open("wb") as file:
                    pickle.dump(
                        cross_domain_results,
                        file,
                        protocol=pickle.HIGHEST_PROTOCOL,
                    )

                

