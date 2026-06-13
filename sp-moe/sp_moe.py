"""
SP_MOE: soft-weighted mixture of experts for spatiotemporally heterogeneous
environmental processes (e.g. wildfire risk over a spatiotemporal grid).

pipeline (driven by .fit()):
  1. train a global base model on all data
  2. cluster spatiotemporal/ecological features with k-means (and optionally
     fit a gmm for posterior-based weighting)
  3. warm-start k specialist models from the global model's weights and
     fine-tune each specialist on its own cluster's subset of data
  4. at inference (.predict()), blend the k specialist outputs using either
     inverse-distance-to-centroid weights or gmm posterior weights computed
     from the cluster features

design notes:
  - cluster_features should generally be ecological/meteorological features
    (e.g. fwi, rh, t, vhi_ave), not raw lat/lon, to avoid arbitrary spatial
    boundaries and to keep the gating mechanism physically interpretable.
  - the global model is used for warm-starting only by default. it is not
    part of the blended prediction, but is exposed via predict_global() for
    ablations (e.g. "global only" condition).
  - predict_expert() exposes a single specialist with no blending, useful
    for the "hard-gated, no warm start" / "hard-gated with warm start"
    ablation conditions.
"""

import copy
import pickle
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist


class SP_MOE(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        k: int = 4,
        global_hidden_dims: Sequence[int] = (64, 64),
        expert_hidden_dims: Sequence[int] = (32, 32),
        weighting: str = "inverse_distance",
        distance_power: float = 2.0,
        eps: float = 1e-6,
        gmm_covariance_type: str = "full",
        random_state: int = 42,
        device: Optional[str] = None,
    ):
        super().__init__()

        if weighting not in ("inverse_distance", "gmm"):
            raise ValueError('weighting must be "inverse_distance" or "gmm"')

        # hyperparameters, all set at construction time so the modules below
        # can actually be built. nothing here is deferred to a later step.
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.k = k
        self.global_hidden_dims = tuple(global_hidden_dims)
        self.expert_hidden_dims = tuple(expert_hidden_dims)
        self.weighting = weighting
        self.distance_power = distance_power
        self.eps = eps
        self.gmm_covariance_type = gmm_covariance_type
        self.random_state = random_state
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # core networks
        self.global_model = self._build_mlp(input_dim, self.global_hidden_dims, output_dim)
        self.experts = nn.ModuleList(
            [self._build_mlp(input_dim, self.expert_hidden_dims, output_dim) for _ in range(k)]
        )
        self.to(self.device)

        # filled in by fit(), left as none until then
        self.cluster_scaler: Optional[StandardScaler] = None
        self.kmeans: Optional[KMeans] = None
        self.gmm: Optional[GaussianMixture] = None
        self.is_fitted = False

    @staticmethod
    def _build_mlp(input_dim: int, hidden_dims: Sequence[int], output_dim: int) -> nn.Sequential:
        # simple relu mlp, one linear + relu per hidden dim, plus an output head
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            prev_dim = h
        layers.append(nn.Linear(prev_dim, output_dim))
        return nn.Sequential(*layers)

    def _to_tensor(self, arr, dtype=torch.float32) -> torch.Tensor:
        if isinstance(arr, torch.Tensor):
            return arr.to(self.device, dtype=dtype)
        return torch.as_tensor(np.asarray(arr), dtype=dtype, device=self.device)

    # ------------------------------------------------------------------
    # fitting
    # ------------------------------------------------------------------

    def fit(
        self,
        X,
        y,
        cluster_features=None,
        epochs_global: int = 100,
        epochs_expert: int = 50,
        lr: float = 1e-3,
        batch_size: int = 256,
        min_cluster_samples: int = 5,
        verbose: bool = True,
    ) -> "SP_MOE":
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        # default to using the model inputs themselves for clustering if no
        # separate spatiotemporal/ecological feature set is given
        if cluster_features is None:
            cluster_features = X
        else:
            cluster_features = np.asarray(cluster_features, dtype=np.float32)

        if verbose:
            print("[sp_moe] stage 1/3: training global model")
        self._fit_global(X, y, epochs_global, lr, batch_size, verbose)

        if verbose:
            print("[sp_moe] stage 2/3: clustering")
        cluster_labels = self._fit_clusters(cluster_features, verbose)

        if verbose:
            print("[sp_moe] stage 3/3: warm-starting and fine-tuning experts")
        self._fit_specialists(X, y, cluster_labels, epochs_expert, lr, batch_size, min_cluster_samples, verbose)

        self.is_fitted = True
        return self

    def _fit_global(self, X, y, epochs, lr, batch_size, verbose):
        self._train_module(self.global_model, X, y, epochs, lr, batch_size, verbose, tag="global")

    def _fit_clusters(self, cluster_features, verbose):
        self.cluster_scaler = StandardScaler()
        scaled = self.cluster_scaler.fit_transform(cluster_features)

        self.kmeans = KMeans(n_clusters=self.k, random_state=self.random_state, n_init=10)
        cluster_labels = self.kmeans.fit_predict(scaled)

        # gmm is fit on the same scaled features so its posteriors live on
        # the same footing as the kmeans centroids
        if self.weighting == "gmm":
            self.gmm = GaussianMixture(
                n_components=self.k,
                covariance_type=self.gmm_covariance_type,
                random_state=self.random_state,
            )
            self.gmm.fit(scaled)

        if verbose:
            counts = np.bincount(cluster_labels, minlength=self.k)
            print(f"[sp_moe] cluster sizes: {counts.tolist()}")

        return cluster_labels

    def _fit_specialists(self, X, y, cluster_labels, epochs, lr, batch_size, min_cluster_samples, verbose):
        for i, expert in enumerate(self.experts):
            # warm start: copy the global model's weights into this expert
            expert.load_state_dict(copy.deepcopy(self.global_model.state_dict()))

            mask = cluster_labels == i
            n_samples = int(mask.sum())

            if n_samples < min_cluster_samples:
                if verbose:
                    print(f"[sp_moe] expert {i}: only {n_samples} samples, keeping global weights")
                continue

            if verbose:
                print(f"[sp_moe] expert {i}: fine-tuning on {n_samples} samples")
            self._train_module(expert, X[mask], y[mask], epochs, lr, batch_size, verbose=False, tag=f"expert_{i}")

    def _train_module(self, module, X, y, epochs, lr, batch_size, verbose, tag=""):
        X_t = self._to_tensor(X)
        y_t = self._to_tensor(y)

        dataset = TensorDataset(X_t, y_t)
        loader = DataLoader(dataset, batch_size=min(batch_size, len(dataset)), shuffle=True)

        optimizer = torch.optim.Adam(module.parameters(), lr=lr)
        loss_fn = nn.MSELoss()

        module.train()
        for epoch in range(epochs):
            running_loss = 0.0
            for xb, yb in loader:
                optimizer.zero_grad()
                pred = module(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * xb.size(0)

            if verbose and (epoch == 0 or (epoch + 1) % max(1, epochs // 5) == 0):
                avg_loss = running_loss / len(dataset)
                print(f"[sp_moe] {tag} epoch {epoch + 1}/{epochs} mse={avg_loss:.6f}")

        module.eval()

    # ------------------------------------------------------------------
    # weighting / inference
    # ------------------------------------------------------------------

    def _compute_weights(self, cluster_features) -> np.ndarray:
        scaled = self.cluster_scaler.transform(cluster_features)

        if self.weighting == "gmm":
            weights = self.gmm.predict_proba(scaled)
        else:
            # inverse-distance weighting from each point to every centroid,
            # normalized so each row sums to 1
            dists = cdist(scaled, self.kmeans.cluster_centers_)
            inv = 1.0 / (dists ** self.distance_power + self.eps)
            weights = inv / inv.sum(axis=1, keepdims=True)

        return weights

    def predict(self, X, cluster_features=None, return_weights: bool = False):
        if not self.is_fitted:
            raise RuntimeError("call .fit() before .predict()")

        X = np.asarray(X, dtype=np.float32)
        if cluster_features is None:
            cluster_features = X
        else:
            cluster_features = np.asarray(cluster_features, dtype=np.float32)

        weights = self._compute_weights(cluster_features)  # shape (n, k)
        weights_t = self._to_tensor(weights)

        X_t = self._to_tensor(X)

        with torch.no_grad():
            expert_preds = torch.stack([expert(X_t) for expert in self.experts], dim=1)  # (n, k, out)
            blended = (weights_t.unsqueeze(-1) * expert_preds).sum(dim=1)  # (n, out)

        blended_np = blended.cpu().numpy()

        if return_weights:
            return blended_np, weights
        return blended_np

    def predict_global(self, X) -> np.ndarray:
        # global-only prediction, useful for the "global only" ablation condition
        X_t = self._to_tensor(np.asarray(X, dtype=np.float32))
        with torch.no_grad():
            pred = self.global_model(X_t)
        return pred.cpu().numpy()

    def predict_expert(self, X, expert_idx: int) -> np.ndarray:
        # single-expert prediction, no blending. useful for hard-gated ablations
        X_t = self._to_tensor(np.asarray(X, dtype=np.float32))
        with torch.no_grad():
            pred = self.experts[expert_idx](X_t)
        return pred.cpu().numpy()

    def assign_clusters(self, cluster_features) -> np.ndarray:
        # hard cluster assignment for new data, useful for hard-gated ablations
        if self.kmeans is None:
            raise RuntimeError("call .fit() before assign_clusters()")
        scaled = self.cluster_scaler.transform(np.asarray(cluster_features, dtype=np.float32))
        return self.kmeans.predict(scaled)

    # ------------------------------------------------------------------
    # persistence
    # ------------------------------------------------------------------

    def save(self, path: str):
        torch.save(
            {
                "state_dict": self.state_dict(),
                "kmeans": pickle.dumps(self.kmeans),
                "gmm": pickle.dumps(self.gmm),
                "cluster_scaler": pickle.dumps(self.cluster_scaler),
                "is_fitted": self.is_fitted,
                "config": {
                    "input_dim": self.input_dim,
                    "output_dim": self.output_dim,
                    "k": self.k,
                    "global_hidden_dims": self.global_hidden_dims,
                    "expert_hidden_dims": self.expert_hidden_dims,
                    "weighting": self.weighting,
                    "distance_power": self.distance_power,
                    "eps": self.eps,
                    "gmm_covariance_type": self.gmm_covariance_type,
                    "random_state": self.random_state,
                },
            },
            path,
        )

    @classmethod
    def load(cls, path: str, device: Optional[str] = None) -> "SP_MOE":
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        config = checkpoint["config"]
        model = cls(**config, device=device)
        model.load_state_dict(checkpoint["state_dict"])
        model.kmeans = pickle.loads(checkpoint["kmeans"])
        model.gmm = pickle.loads(checkpoint["gmm"])
        model.cluster_scaler = pickle.loads(checkpoint["cluster_scaler"])
        model.is_fitted = checkpoint["is_fitted"]
        model.to(model.device)
        return model