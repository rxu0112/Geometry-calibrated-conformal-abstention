"""Class-conditional distance summaries for geometry calibration.

All estimators expose the same interface:

    distance_model = ClassConditionalDistance(metric="shrinkage_mahalanobis")
    distance_model.fit(correct_reference, incorrect_reference)
    distance_to_correct, distance_to_incorrect = distance_model.transform_one(x)

This allows the distance metric to change without changing the downstream
calibrator or conformal evaluation.
"""

from dataclasses import dataclass

import numpy as np
import torch
from sklearn.covariance import LedoitWolf
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


SUPPORTED_METRICS = (
    "empirical_mahalanobis",
    "shrinkage_mahalanobis",
    "diagonal_mahalanobis",
    "standardized_euclidean",
    "knn",
)


@dataclass
class _MahalanobisState:
    mean: np.ndarray
    precision: np.ndarray


@dataclass
class _DiagonalState:
    mean: np.ndarray
    variance: np.ndarray


class ClassConditionalDistance:
    """Fit matching distance summaries for correct and incorrect references."""

    def __init__(
        self,
        metric="shrinkage_mahalanobis",
        eps=1e-6,
        n_neighbors=10,
        knn_device="auto",
        knn_batch_size=1024,
        knn_n_jobs=-1,
    ):
        if metric not in SUPPORTED_METRICS:
            raise ValueError(
                f"Unsupported metric {metric!r}; choose from {SUPPORTED_METRICS}."
            )
        if eps <= 0:
            raise ValueError("eps must be positive.")
        if n_neighbors < 1:
            raise ValueError("n_neighbors must be at least 1.")
        if knn_batch_size < 1:
            raise ValueError("knn_batch_size must be at least 1.")

        self.metric = metric
        self.eps = eps
        self.n_neighbors = n_neighbors
        self.knn_device = knn_device
        self.knn_batch_size = knn_batch_size
        self.knn_n_jobs = knn_n_jobs
        self.correct_state = None
        self.incorrect_state = None
        self.scaler = None
        self._knn_uses_torch = False

    @staticmethod
    def _validate_matrix(X, name):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError(f"{name} must be a two-dimensional array.")
        if len(X) < 2:
            raise ValueError(f"{name} must contain at least two samples.")
        if not np.all(np.isfinite(X)):
            raise ValueError(f"{name} contains NaN or infinite values.")
        return X

    def fit(self, correct_reference, incorrect_reference):
        correct = self._validate_matrix(correct_reference, "correct_reference")
        incorrect = self._validate_matrix(
            incorrect_reference, "incorrect_reference"
        )
        if correct.shape[1] != incorrect.shape[1]:
            raise ValueError("Correct and incorrect references must have equal width.")

        if self.metric == "empirical_mahalanobis":
            self.correct_state = self._fit_empirical_mahalanobis(correct)
            self.incorrect_state = self._fit_empirical_mahalanobis(incorrect)
        elif self.metric == "shrinkage_mahalanobis":
            self.correct_state = self._fit_shrinkage_mahalanobis(correct)
            self.incorrect_state = self._fit_shrinkage_mahalanobis(incorrect)
        elif self.metric == "diagonal_mahalanobis":
            self.correct_state = self._fit_diagonal(correct)
            self.incorrect_state = self._fit_diagonal(incorrect)
        else:
            # Euclidean and k-NN use one label-agnostic transformation fitted on
            # the combined reference split, ensuring both distances share a scale.
            self.scaler = StandardScaler().fit(np.vstack([correct, incorrect]))
            correct_scaled = self.scaler.transform(correct).astype(
                np.float32, copy=False
            )
            incorrect_scaled = self.scaler.transform(incorrect).astype(
                np.float32, copy=False
            )

            if self.metric == "standardized_euclidean":
                self.correct_state = correct_scaled.mean(axis=0)
                self.incorrect_state = incorrect_scaled.mean(axis=0)
            else:
                correct_k = min(self.n_neighbors, len(correct_scaled))
                incorrect_k = min(self.n_neighbors, len(incorrect_scaled))
                device = self._resolve_knn_device()
                self._knn_uses_torch = device.type == "cuda"

                if self._knn_uses_torch:
                    self.correct_state = (
                        torch.as_tensor(correct_scaled, device=device),
                        correct_k,
                    )
                    self.incorrect_state = (
                        torch.as_tensor(incorrect_scaled, device=device),
                        incorrect_k,
                    )
                else:
                    self.correct_state = NearestNeighbors(
                        n_neighbors=correct_k,
                        metric="euclidean",
                        n_jobs=self.knn_n_jobs,
                    ).fit(correct_scaled)
                    self.incorrect_state = NearestNeighbors(
                        n_neighbors=incorrect_k,
                        metric="euclidean",
                        n_jobs=self.knn_n_jobs,
                    ).fit(incorrect_scaled)

        return self

    def _resolve_knn_device(self):
        if self.knn_device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device = torch.device(self.knn_device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA k-NN requested, but CUDA is not available.")
        return device

    def _torch_knn_distance(self, X, state):
        reference, n_neighbors = state
        output = []

        with torch.inference_mode():
            for start in range(0, len(X), self.knn_batch_size):
                stop = min(start + self.knn_batch_size, len(X))
                query = torch.as_tensor(
                    X[start:stop],
                    dtype=reference.dtype,
                    device=reference.device,
                )
                pairwise_distance = torch.cdist(query, reference)
                nearest_distance = torch.topk(
                    pairwise_distance,
                    k=n_neighbors,
                    dim=1,
                    largest=False,
                    sorted=False,
                ).values
                output.append(nearest_distance.mean(dim=1).cpu().numpy())

        return np.concatenate(output)

    def _fit_empirical_mahalanobis(self, X):
        mean = X.mean(axis=0)
        covariance = np.atleast_2d(np.cov(X, rowvar=False))
        covariance += self.eps * np.eye(covariance.shape[0])
        # pinv is safer than inv when the empirical covariance is ill-conditioned.
        precision = np.linalg.pinv(covariance, hermitian=True)
        return _MahalanobisState(mean, precision)

    @staticmethod
    def _fit_shrinkage_mahalanobis(X):
        estimator = LedoitWolf().fit(X)
        return _MahalanobisState(estimator.location_, estimator.precision_)

    def _fit_diagonal(self, X):
        return _DiagonalState(
            mean=X.mean(axis=0),
            variance=X.var(axis=0, ddof=1) + self.eps,
        )

    @staticmethod
    def _mahalanobis(X, state):
        difference = X - state.mean
        squared = np.einsum(
            "ni,ij,nj->n",
            difference,
            state.precision,
            difference,
        )
        return np.sqrt(np.maximum(squared, 0.0))

    @staticmethod
    def _diagonal_mahalanobis(X, state):
        squared = np.sum((X - state.mean) ** 2 / state.variance, axis=1)
        return np.sqrt(np.maximum(squared, 0.0))

    def transform(self, X):
        if self.correct_state is None or self.incorrect_state is None:
            raise RuntimeError("Call fit before transform.")

        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.ndim != 2 or not np.all(np.isfinite(X)):
            raise ValueError("X must be a finite one- or two-dimensional array.")

        if self.metric in (
            "empirical_mahalanobis",
            "shrinkage_mahalanobis",
        ):
            correct_distance = self._mahalanobis(X, self.correct_state)
            incorrect_distance = self._mahalanobis(X, self.incorrect_state)
        elif self.metric == "diagonal_mahalanobis":
            correct_distance = self._diagonal_mahalanobis(
                X, self.correct_state
            )
            incorrect_distance = self._diagonal_mahalanobis(
                X, self.incorrect_state
            )
        else:
            X_scaled = self.scaler.transform(X).astype(np.float32, copy=False)
            if self.metric == "standardized_euclidean":
                correct_distance = np.linalg.norm(
                    X_scaled - self.correct_state, axis=1
                )
                incorrect_distance = np.linalg.norm(
                    X_scaled - self.incorrect_state, axis=1
                )
            elif self._knn_uses_torch:
                correct_distance = self._torch_knn_distance(
                    X_scaled, self.correct_state
                )
                incorrect_distance = self._torch_knn_distance(
                    X_scaled, self.incorrect_state
                )
            else:
                correct_neighbors = self.correct_state.kneighbors(
                    X_scaled, return_distance=True
                )[0]
                incorrect_neighbors = self.incorrect_state.kneighbors(
                    X_scaled, return_distance=True
                )[0]
                correct_distance = correct_neighbors.mean(axis=1)
                incorrect_distance = incorrect_neighbors.mean(axis=1)

        return np.column_stack([correct_distance, incorrect_distance])

    def transform_one(self, x):
        distances = self.transform(x)[0]
        return float(distances[0]), float(distances[1])
