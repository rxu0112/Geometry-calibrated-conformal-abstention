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

device = 'cuda:0'
CONF_ONLY = True
for dataset in ['sciq', 'nq', 'common', 'simple_questions_wiki', 'truthfulQA', 'gsm8k']: # 'sciq', 'nq', 'common', 'simple_questions_wiki', 'truthfulQA', 'gsm8k'
    for model in ['Llama-3.2-3B-Instruct', 'Gemma-3-4B-it', 'Llama-3-8B-Instruct']: # 'Llama-3.2-3B-Instruct', 'Gemma-3-4B-it', 'Llama-3-8B-Instruct'

        statistic_results = torch.load(f"results/{model}/{dataset}/{dataset}_pca_features.pt", weights_only=False)
        hem_examples = statistic_results['hem_examples']
        prob_examples = statistic_results['prob_examples']
        pca_hidden_examples = statistic_results['pca_hidden_examples']
        ref_idx = statistic_results['ref_idx']
        remain_idx = statistic_results['remain_idx']

        right_threshold = 0.8
        wrong_threshold = 0.2
        y = np.array([1 if s > right_threshold else 0 for s in hem_examples])
        remain_y = y[remain_idx]
        remain_prob_examples = [prob_examples[r] for r in remain_idx]


        right_ref_pca_examples = [
            pca_hidden_examples[r]
            for r in ref_idx
            if hem_examples[r] > right_threshold
        ]
        print(len(right_ref_pca_examples))
        wrong_ref_pca_examples = [
            pca_hidden_examples[r]
            for r in ref_idx
            if hem_examples[r] < wrong_threshold
        ]

        # Each stored response has shape (2L-1, num_tokens). Pool its token
        # columns to obtain (num_reference_tokens, 2L-1).
        right_ref_metric_distribution = np.column_stack(
            right_ref_pca_examples
        ).T
        wrong_ref_metric_distribution = np.column_stack(
            wrong_ref_pca_examples
        ).T

        right_mu, right_Sigma_inv = precompute_mahalanobis(right_ref_metric_distribution)
        wrong_mu, wrong_Sigma_inv = precompute_mahalanobis(wrong_ref_metric_distribution)

        remain_pca_hidden_examples = [pca_hidden_examples[r] for r in remain_idx]

        coordinates = []
        for s_idx in tqdm(range(len(remain_idx))):
            distance_to_right = []
            distance_to_wrong = []
            for t_idx in range(len(remain_prob_examples[s_idx])):
                pca_vector = remain_pca_hidden_examples[s_idx][:, t_idx]
                distance_to_right.append(mahalanobis_distance(pca_vector, right_mu, right_Sigma_inv))
                distance_to_wrong.append(mahalanobis_distance(pca_vector, wrong_mu, wrong_Sigma_inv))
            coords = np.stack([distance_to_right, distance_to_wrong], axis=1)
            coordinates.append(coords)

        # Calibrator Training
        features = []
        if CONF_ONLY:
            for coords, prob in zip(coordinates, remain_prob_examples):
                prob = prob.reshape(-1, 1)   # (n, 1)
                features.append(np.concatenate([prob], axis=1))
        else:
            for coords, prob in zip(coordinates, remain_prob_examples):
                prob = prob.reshape(-1, 1)   # (n, 1)
                features.append(np.concatenate([coords,prob], axis=1))

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

        dtrain = xgb.DMatrix(X_train, label=y_train_broadcast)
        dval = xgb.DMatrix(X_val, label=y_val_broadcast)
        dtest = xgb.DMatrix(X_test, label=y_test_broadcast)

        def objective(trial):
            params = {
                "objective": "binary:logistic",
                "eval_metric": "auc",
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

            val_auc = roc_auc_score(y_val_broadcast, y_proba_val)
            val_auprc = average_precision_score(y_val_broadcast, y_proba_val)
            score = val_auc+val_auprc
                
            # You can store best_round in trial user attributes
            trial.set_user_attr("best_round", best_round)

            return score

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=50)

        print("Best validation AUROC+AUPRC:", study.best_value)
        print("Best params:", study.best_params)

        dtrain_full = xgb.DMatrix(np.vstack([X_train, X_val]), label=np.hstack([y_train_broadcast, y_val_broadcast]))
        final_params = {
            'device':device,
            **study.best_params,
        }
        bst = xgb.train(
            params=final_params,
            dtrain=dtrain_full,
            num_boost_round=study.best_trial.user_attrs["best_round"] + 1,
            verbose_eval=False
        )
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

        y_proba_train_broadcast = bst.predict(dtrain)
        y_proba_train = []
        start = 0
        for feats in train_features:
            n = len(feats)
            seq = y_proba_train_broadcast[start:start + n]
            y_proba_train.append(np.log(np.clip(seq, 1e-12, 1.0)).mean())
            start += n

        auprc_train = average_precision_score(y_train, y_proba_train)
        print('train auprc:', auprc_train)
        auroc_train = roc_auc_score(y_train, y_proba_train)
        print('train auroc:', auroc_train)

        if CONF_ONLY:
            np.save(f"results/{model}/{dataset}/test/label_conf_only.npy", y_test)
            np.save(f"results/{model}/{dataset}/test/proposed_conf_only.npy", np.array(y_proba_test))
        else:
            np.save(f"results/{model}/{dataset}/test/label_pca.npy", y_test)
            np.save(f"results/{model}/{dataset}/test/proposed_pca.npy", np.array(y_proba_test))
