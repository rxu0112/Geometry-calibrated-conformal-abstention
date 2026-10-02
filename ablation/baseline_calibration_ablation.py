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

for baseline_metric in ['AttentionScore', 'Perplexity', 'SemanticEntropy', 'PTrue', 'SAR', 'EigenScore', 'Focus']:
    for dataset in ['common','nq','truthfulQA','sciq','simple_questions_wiki','gsm8k']:
        for model in ['Llama-3-8B-Instruct','Gemma-3-4B-it','Llama-3.2-3B-Instruct']:
            all_labels = torch.load(f"processed_data/{dataset}/{model}_labels.pt", weights_only=False)

            y_train = all_labels['y_train']
            y_val = all_labels['y_val']
            y_test = all_labels['y_test'] # this equals to np.load(f"results/{model}/{dataset}/test/label.npy")

            X_train = np.load(f"results/{model}/{dataset}/train/{baseline_metric}.npy").reshape(-1, 1)
            X_val = np.load(f"results/{model}/{dataset}/val/{baseline_metric}.npy").reshape(-1, 1)
            X_test = np.load(f"results/{model}/{dataset}/test/{baseline_metric}.npy").reshape(-1, 1)

            dtrain = xgb.DMatrix(X_train, label=y_train)
            dval = xgb.DMatrix(X_val, label=y_val)
            dtest = xgb.DMatrix(X_test, label=y_test)

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

                val_auc = roc_auc_score(y_val, y_proba_val)
                val_auprc = average_precision_score(y_val, y_proba_val)
                score = val_auc+val_auprc
                    
                # You can store best_round in trial user attributes
                trial.set_user_attr("best_round", best_round)

                return score


            study = optuna.create_study(direction="maximize")
            study.optimize(objective, n_trials=50)

            print("Best validation AUROC:", study.best_value)
            print("Best params:", study.best_params)

            dtrain_full = xgb.DMatrix(np.vstack([X_train, X_val]), label=np.hstack([y_train, y_val]))
            bst = xgb.train(
                params=study.best_params,
                dtrain=dtrain_full,
                num_boost_round=study.best_trial.user_attrs["best_round"],
                verbose_eval=False
            )

            y_proba_test = bst.predict(dtest)
            np.save(f"results/{model}/{dataset}/test/{baseline_metric}_calibrated.npy", np.array(y_proba_test))




