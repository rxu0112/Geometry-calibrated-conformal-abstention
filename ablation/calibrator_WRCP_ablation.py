import copy
import random

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from general_utils import *
from tqdm import tqdm

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def wasserstein_1d(x, y, num_quantiles=128):
    """Differentiable quantile approximation of the one-dimensional W1 distance."""
    if x.numel() == 0 or y.numel() == 0:
        return x.new_tensor(0.0)

    num_quantiles = min(num_quantiles, x.numel(), y.numel())
    q = (torch.arange(num_quantiles, device=x.device, dtype=x.dtype) + 0.5)
    q = q / num_quantiles
    return torch.mean(torch.abs(torch.quantile(x, q) - torch.quantile(y, q)))


def class_conditional_wasserstein(prob_1, y_1, prob_2, y_2):
    """Align residual distributions separately for incorrect and correct labels."""
    residual_1 = torch.abs(prob_1 - y_1)
    residual_2 = torch.abs(prob_2 - y_2)

    distances = []
    for label in (0.0, 1.0):
        r_1 = residual_1[y_1 == label]
        r_2 = residual_2[y_2 == label]
        if r_1.numel() > 0 and r_2.numel() > 0:
            distances.append(wasserstein_1d(r_1, r_2))

    if not distances:
        return prob_1.new_tensor(0.0)
    return torch.stack(distances).mean()


class WassersteinMLPCalibrator:
    def __init__(
        self,
        hidden_dim=64,
        dropout=0.1,
        lambda_w=1.0,
        learning_rate=1e-3,
        weight_decay=1e-4,
        max_epochs=1000,
        patience=50,
        validation_fraction=0.2,
        random_state=42,
        device=None,
    ):
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.lambda_w = lambda_w
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.patience = patience
        self.validation_fraction = validation_fraction
        self.random_state = random_state
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.scaler = StandardScaler()
        self.model = None

    def _split(self, X, y):
        return train_test_split(
            np.asarray(X, dtype=np.float32),
            np.asarray(y, dtype=np.float32),
            test_size=self.validation_fraction,
            random_state=self.random_state,
            stratify=y,
        )

    def _tensor(self, array):
        return torch.as_tensor(array, dtype=torch.float32, device=self.device)

    def _objective(self, X_1, y_1, X_2, y_2):
        logits_1 = self.model(X_1)
        logits_2 = self.model(X_2)

        # Averaging the two domain losses prevents the larger source from dominating.
        supervised_loss = 0.5 * (
            nn.functional.binary_cross_entropy_with_logits(logits_1, y_1)
            + nn.functional.binary_cross_entropy_with_logits(logits_2, y_2)
        )

        prob_1 = torch.sigmoid(logits_1)
        prob_2 = torch.sigmoid(logits_2)
        wasserstein_loss = class_conditional_wasserstein(prob_1, y_1, prob_2, y_2)
        total_loss = supervised_loss + self.lambda_w * wasserstein_loss
        return total_loss, supervised_loss, wasserstein_loss

    def fit(self, X_source_1, y_source_1, X_source_2, y_source_2, verbose=True):
        random.seed(self.random_state)
        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_state)

        X1_train, X1_val, y1_train, y1_val = self._split(X_source_1, y_source_1)
        X2_train, X2_val, y2_train, y2_val = self._split(X_source_2, y_source_2)

        # Fit normalization using only model-training examples, never CP calibration data.
        self.scaler.fit(np.concatenate([X1_train, X2_train], axis=0))
        X1_train = self.scaler.transform(X1_train)
        X2_train = self.scaler.transform(X2_train)
        X1_val = self.scaler.transform(X1_val)
        X2_val = self.scaler.transform(X2_val)

        train_tensors = tuple(
            self._tensor(x) for x in (X1_train, y1_train, X2_train, y2_train)
        )
        val_tensors = tuple(
            self._tensor(x) for x in (X1_val, y1_val, X2_val, y2_val)
        )

        self.model = MLP(
            input_dim=X1_train.shape[1],
            hidden_dim=self.hidden_dim,
            dropout=self.dropout,
        ).to(self.device)
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        best_state = None
        best_val_loss = float("inf")
        epochs_without_improvement = 0

        for epoch in range(self.max_epochs):
            self.model.train()
            train_loss, train_bce, train_w = self._objective(*train_tensors)
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            self.model.eval()
            with torch.no_grad():
                val_loss, val_bce, val_w = self._objective(*val_tensors)
            val_value = val_loss.item()

            if val_value < best_val_loss - 1e-6:
                best_val_loss = val_value
                best_state = copy.deepcopy(self.model.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if verbose and (epoch == 0 or (epoch + 1) % 50 == 0):
                print(
                    f"epoch={epoch + 1:4d} "
                    f"train={train_loss.item():.5f} "
                    f"bce={train_bce.item():.5f} "
                    f"wasserstein={train_w.item():.5f} "
                    f"val={val_value:.5f}"
                )

            if epochs_without_improvement >= self.patience:
                break

        self.model.load_state_dict(best_state)
        self.model.eval()
        return self

    def predict_proba(self, X):
        if self.model is None:
            raise RuntimeError("Call fit before predict_proba.")

        X = np.asarray(X, dtype=np.float32)
        X = self.scaler.transform(X)
        with torch.no_grad():
            logits = self.model(self._tensor(X))
            probabilities = torch.sigmoid(logits)
        return probabilities.cpu().numpy()


X_train_list = []
y_train_list = []
y_train_broadcast_list = []
X_test_list = []
y_test_list = []
y_test_broadcast_list = []
test_features_list = []
source_names = []

for dataset in ['simple_questions_wiki','nq']:
    
    model = 'Gemma-3-4B-it'
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
    scaler.fit(X)

    X_train = scaler.transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)

    X_train_list.append(X_train)
    y_train_list.append(y_train)
    y_train_broadcast_list.append(y_train_broadcast)
    X_test_list.append(X_test)
    y_test_list.append(y_test)
    y_test_broadcast_list.append(y_test_broadcast)
    test_features_list.append(test_features)
    source_names.append(f'{dataset}_{model}')


X_train_1 = X_train_list[0]
X_train_2 = X_train_list[1]
y_train_broadcast_1 = y_train_broadcast_list[0]
y_train_broadcast_2 = y_train_broadcast_list[1]
y_train_1 = y_train_list[0]
y_train_2 = y_train_list[1]

X_test_1 = X_test_list[0]
X_test_2 = X_test_list[1]
y_test_broadcast_1 = y_test_broadcast_list[0]
y_test_broadcast_2 = y_test_broadcast_list[1]
y_test_1 = y_test_list[0]
y_test_2 = y_test_list[1]

test_features_1 = test_features_list[0]
test_features_2 = test_features_list[1]

calibrator = WassersteinMLPCalibrator(
    hidden_dim=64,
    lambda_w=0.03,
    learning_rate=1e-3,
    max_epochs=2000,
    patience=100,
)

calibrator.fit(
    X_train_1,
    y_train_broadcast_1,
    X_train_2,
    y_train_broadcast_2,
)

y_proba_test_broadcast_1 = calibrator.predict_proba(X_test_1)
y_proba_test_broadcast_2 = calibrator.predict_proba(X_test_2)


y_proba_test_1 = []
start = 0
for feats in test_features_1:
    n = len(feats)
    seq = y_proba_test_broadcast_1[start:start + n]
    y_proba_test_1.append(np.log(np.clip(seq, 1e-12, 1.0)).mean())
    start += n

y_proba_test_2 = []
start = 0
for feats in test_features_2:
    n = len(feats)
    seq = y_proba_test_broadcast_2[start:start + n]
    y_proba_test_2.append(np.log(np.clip(seq, 1e-12, 1.0)).mean())
    start += n

y_cal = y_test_1
y_test = y_test_2

s_cal = - np.array(y_proba_test_1)
s_test = - np.array(y_proba_test_2)
s_cal_sorted = np.sort(s_cal)

C_cal = np.where(y_cal == 1)[0]
I_cal = np.where(y_cal == 0)[0]
s_cal_corr = s_cal[C_cal]
s_cal_inc = s_cal[I_cal]
s_cal_corr_sorted = np.sort(s_cal_corr)
s_cal_inc_sorted = np.sort(s_cal_inc)

# --- empirical conditional probability on test
C_test = np.where(y_test == 1)[0]
I_test = np.where(y_test == 0)[0]
s_test_corr = s_test[C_test]
s_test_inc = s_test[I_test]

def cp_threshold(scores_sorted, quantiles):
    """Return the empirical CP threshold (1-alpha coverage)"""
    n = len(scores_sorted)
    thresholds = []
    for level in quantiles:
        k = int(np.ceil((level) * (n + 1))) - 1  # -1 because Python index starts at 0
        k = min(k, n-1)  # avoid going out of bounds
        thresholds.append(scores_sorted[k])
    return np.array(thresholds)

retained_level = np.arange(0.1, 1.00, 0.1)
thresholds = cp_threshold(s_cal_sorted, retained_level)

def compute_conditioned_bound(idx,tau):

    # --- compute 1 - beta_corr
    # quantiles of correct-only calibration scores
    quantiles = np.linspace(0, 1, 1001)
    cal_corr_quantiles = cp_threshold(s_cal_corr_sorted, quantiles)
    cal_inc_quantiles = cp_threshold(s_cal_inc_sorted, quantiles)


    mask_corr = cal_corr_quantiles <= tau
    if not np.any(mask_corr):
        one_minus_beta_corr = 0.0
    else:
        one_minus_beta_corr = np.max(quantiles[mask_corr])

    mask_inc = cal_inc_quantiles <= tau
    if not np.any(mask_inc):
        one_minus_beta_inc = 0.0
    else:
        one_minus_beta_inc = np.max(quantiles[mask_inc])

    # theoretical interval
    lower_bound_corr = one_minus_beta_corr

    lower_bound_corr = np.clip(lower_bound_corr, 0, 1)
    lower_bound_all = retained_level[idx]
    upper_bound_all = lower_bound_all + 1.0 / (len(s_cal) + 1)
    lower_bound_cond_corr = lower_bound_corr/upper_bound_all*(len(s_test_corr)+len(s_cal_corr))/(len(s_test)+len(s_cal))
    ratio_all = np.mean(s_test <= tau)


    if (np.sum(s_test_corr <= tau)+np.sum(s_test_inc <= tau))==0:
        prob_cond_corr = 0.0
    else:
        prob_cond_corr = np.sum(s_test_corr <= tau)/(np.sum(s_test_corr <= tau)+np.sum(s_test_inc <= tau))
    
    lower_bounds_all.append(lower_bound_all)
    lower_bounds_cond_corr.append(lower_bound_cond_corr)
    probs_cond_corr.append(prob_cond_corr)
    ratios_all.append(ratio_all)

lower_bounds_all=[]
lower_bounds_cond_corr=[]
probs_cond_corr = []
ratios_all = []
for idx, tau in enumerate(thresholds):
    compute_conditioned_bound(idx, tau)

print(lower_bounds_all[4])
print(ratios_all[4])

print(lower_bounds_cond_corr[4])
print(probs_cond_corr[4])

print('finished')

