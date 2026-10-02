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

def build_metric_distribution(example_list, selected_groups):
    parts = []
    for g in selected_groups:
        parts.append(np.column_stack(example_list[g]).T)    
    return np.concatenate(parts, axis=1)

for dataset in ['nq', 'sciq', 'simple_questions_wiki', 'truthfulQA','gsm8k','common']: # 'sciq', 'nq', 'simple_questions_wiki', 'truthfulQA'
    for model in ['Llama-3.2-3B-Instruct','Gemma-3-4B-it','Llama-3-8B-Instruct']: # 'Gemma-3-4B-it','Llama-3.2-3B-Instruct'

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

        indices = np.arange(len(remain_y))

        temp_idx, test_idx, y_temp, y_test = train_test_split(
            indices, remain_y, test_size=0.2, random_state=42, stratify=remain_y)

        train_idx, val_idx, y_train, y_val = train_test_split(
            temp_idx, y_temp, test_size=0.25, random_state=42, stratify=y_temp)

        right_ref_contri_examples = [contri_examples[r] for r in ref_idx if hem_examples[r] > right_threshold]
        wrong_ref_contri_examples = [contri_examples[r] for r in ref_idx if hem_examples[r] < wrong_threshold]
        right_ref_angle_diff_examples = [angle_diff_examples[r] for r in ref_idx if hem_examples[r] > right_threshold]
        wrong_ref_angle_diff_examples = [angle_diff_examples[r] for r in ref_idx if hem_examples[r] < wrong_threshold]
        right_ref_cos_to_right_hidden_avg_examples = [cos_to_right_hidden_avg_examples[r] for r in ref_idx if hem_examples[r] > right_threshold]
        wrong_ref_cos_to_right_hidden_avg_examples = [cos_to_right_hidden_avg_examples[r] for r in ref_idx if hem_examples[r] < wrong_threshold]
        right_ref_cos_to_wrong_hidden_avg_examples = [cos_to_wrong_hidden_avg_examples[r] for r in ref_idx if hem_examples[r] > right_threshold]
        wrong_ref_cos_to_wrong_hidden_avg_examples = [cos_to_wrong_hidden_avg_examples[r] for r in ref_idx if hem_examples[r] < wrong_threshold]
        remain_contri_examples = [contri_examples[r] for r in remain_idx]
        remain_angle_diff_examples = [angle_diff_examples[r] for r in remain_idx]
        remain_cos_to_right_hidden_avg_examples = [cos_to_right_hidden_avg_examples[r] for r in remain_idx]
        remain_cos_to_wrong_hidden_avg_examples = [cos_to_wrong_hidden_avg_examples[r] for r in remain_idx]


        right_ref_dict = {
            "contri": right_ref_contri_examples,
            "angle_diff": right_ref_angle_diff_examples,
            "cos_right": right_ref_cos_to_right_hidden_avg_examples,
            "cos_wrong": right_ref_cos_to_wrong_hidden_avg_examples,
        }

        wrong_ref_dict = {
            "contri": wrong_ref_contri_examples,
            "angle_diff": wrong_ref_angle_diff_examples,
            "cos_right": wrong_ref_cos_to_right_hidden_avg_examples,
            "cos_wrong": wrong_ref_cos_to_wrong_hidden_avg_examples,
        }

        remain_dict = {
            "contri": remain_contri_examples,
            "angle_diff": remain_angle_diff_examples,
            "cos_right": remain_cos_to_right_hidden_avg_examples,
            "cos_wrong": remain_cos_to_wrong_hidden_avg_examples,
        }

        group_names = ["contri", "angle_diff", "cos_right", "cos_wrong"]

        permuted_remain_dict = {}

        for g in group_names:
            # collect all tokens across all samples
            all_tokens = np.concatenate([x.T for x in remain_dict[g]], axis=0)  # (total_tokens, dim)

            perm = np.random.permutation(len(all_tokens))
            all_tokens_perm = all_tokens[perm]

            # now re-split back to original structure
            permuted_list = []
            start = 0

            for x in remain_dict[g]:
                n_tokens = x.shape[1]
                chunk = all_tokens_perm[start:start+n_tokens]   # (n_tokens, dim)
                permuted_list.append(chunk.T)                  # back to (dim, n_tokens)
                start += n_tokens

            permuted_remain_dict[g] = permuted_list

        # ref is based on all metrics
        right_ref_metric_distribution = build_metric_distribution(right_ref_dict, group_names)
        wrong_ref_metric_distribution = build_metric_distribution(wrong_ref_dict, group_names)
        right_mu, right_Sigma_inv = precompute_mahalanobis(right_ref_metric_distribution)
        wrong_mu, wrong_Sigma_inv = precompute_mahalanobis(wrong_ref_metric_distribution)

        # train and val data is based on all metrics
        coordinates = []
        for s_idx in tqdm(range(len(remain_idx))):
            distance_to_right = []
            distance_to_wrong = []

            for t_idx in range(len(remain_prob_examples[s_idx])):
                metric_parts = []
                for g in group_names:
                    metric_parts.append(remain_dict[g][s_idx][:, t_idx])
                metric = np.concatenate(metric_parts)

                distance_to_right.append(mahalanobis_distance(metric, right_mu, right_Sigma_inv))
                distance_to_wrong.append(mahalanobis_distance(metric, wrong_mu, wrong_Sigma_inv))

            coords = np.stack([distance_to_right, distance_to_wrong], axis=1)
            coordinates.append(coords)
        
        features = []
        for coords, prob in zip(coordinates, remain_prob_examples):
            prob = prob.reshape(-1, 1)   # (n, 1)
            features.append(np.concatenate([coords, prob], axis=1))

        X = np.concatenate(features,axis=0)
        scaler = StandardScaler()
        scaler.fit(X)

        train_features = [features[i] for i in train_idx]
        val_features = [features[i] for i in val_idx]

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

        X_train = scaler.transform(X_train)
        X_val   = scaler.transform(X_val)

        dtrain = xgb.DMatrix(X_train, label=y_train_broadcast)
        dval = xgb.DMatrix(X_val, label=y_val_broadcast)

        # Calibrator Training
        def objective(trial, metric = 'auc'):
            params = {
                "objective": "binary:logistic",
                "eval_metric": metric,
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
            val_auc = roc_auc_score(y_val_broadcast, y_proba_val)
            val_auprc = average_precision_score(y_val_broadcast, y_proba_val)
            score = val_auc+val_auprc
                
            trial.set_user_attr("best_round", best_round)

            return score


        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=50)

        print("Best validation AUROC:", study.best_value)
        print("Best params:", study.best_params)

        dtrain_full = xgb.DMatrix(np.vstack([X_train, X_val]), label=np.hstack([y_train_broadcast, y_val_broadcast]))
        bst = xgb.train(
            params=study.best_params,
            dtrain=dtrain_full,
            num_boost_round=study.best_trial.user_attrs["best_round"],
            verbose_eval=False
        )
        
        # inference data is based on permuted metrics
        ablation_results = {}
        for drop_group in group_names + ["none"]:
            if drop_group == "none":
                selected_groups = group_names
            else:
                selected_groups = [g for g in group_names if g != drop_group]

            coordinates = []
            for s_idx in tqdm(range(len(remain_idx))):
                distance_to_right = []
                distance_to_wrong = []

                for t_idx in range(len(remain_prob_examples[s_idx])):
                    metric_parts = []
                    for g in group_names:
                        if g in selected_groups:
                            data = remain_dict[g][s_idx][:, t_idx]
                        else:
                            data = permuted_remain_dict[g][s_idx][:, t_idx]
                        metric_parts.append(data)
                    

                    metric = np.concatenate(metric_parts)

                    distance_to_right.append(mahalanobis_distance(metric, right_mu, right_Sigma_inv))
                    distance_to_wrong.append(mahalanobis_distance(metric, wrong_mu, wrong_Sigma_inv))

                coords = np.stack([distance_to_right, distance_to_wrong], axis=1)
                coordinates.append(coords)
            
            features = []
            for coords, prob in zip(coordinates, remain_prob_examples):
                prob = prob.reshape(-1, 1)   # (n, 1)
                features.append(np.concatenate([coords, prob], axis=1))
            
            test_features = [features[i] for i in test_idx]

            X_test = np.concatenate(test_features, axis=0)
            y_test_broadcast = np.concatenate([
                np.full(len(features[i]), remain_y[i])
                for i in test_idx
            ])

            X_test = scaler.transform(X_test)
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
            auroc_test = roc_auc_score(y_test, y_proba_test)

            ablation_results[drop_group] = {
                "auprc": auprc_test,
                "auroc": auroc_test
            }
    
        save_path = f"results/{model}/{dataset}/test/proposed_permutation.json"
        with open(save_path, "w") as f:
            json.dump(ablation_results, f, indent=4)
        




