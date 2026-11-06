"""
LACE-DR core components: anomaly injection, encoder, decoders and the base model.

The vectorised training and scoring code that produced the results in the paper is in
lacedr/model.py (class LACE_DR_Vec), which extends the base class defined here.
"""

import os
import random
import numpy as np
import tensorflow as tf
from scipy import sparse
from sklearn.metrics import roc_auc_score, average_precision_score

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except Exception as e:
        print("Could not set memory growth:", e)

from tensorflow.keras import layers, Model


# ============================================================
# 0. Reproducibility
# ============================================================

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


# ============================================================
# 1. Metrics
# ============================================================

def compute_auc_ap(scores, labels):
    scores = np.asarray(scores).reshape(-1)
    labels = np.asarray(labels).astype(int).reshape(-1)
    auc = roc_auc_score(labels, scores)
    ap = average_precision_score(labels, scores)
    return auc, ap


# ============================================================
# ============================================================
# 3. Tuned anomaly injection
# ============================================================

def inject_feature_and_structural_anomalies(
    x,
    a_sp,
    ratio: float = 0.05,
    feat_noise_scale: float = 0.62,
    edge_perturb_ratio: float = 0.20,
    feature_swap_ratio: float = 0.10,
    structure_only_ratio: float = 0.08,
):
    """
    Tuned anomaly injection:
    - moderately stronger Gaussian feature anomalies
    - smaller proportion of swap anomalies
    - smaller proportion of structure-only anomalies
    - partial structural rewiring
    """
    print("\n[Injection] ratio =", ratio)
    print("[Injection] feat_noise_scale =", feat_noise_scale)
    print("[Injection] edge_perturb_ratio =", edge_perturb_ratio)
    print("[Injection] feature_swap_ratio =", feature_swap_ratio)
    print("[Injection] structure_only_ratio =", structure_only_ratio, "\n")

    N, F = x.shape
    num_anom = max(1, int(ratio * N))

    if sparse.issparse(a_sp):
        A = a_sp.toarray().astype("float32")
    else:
        A = np.asarray(a_sp, dtype=np.float32)

    np.fill_diagonal(A, 0.0)
    A = (A > 0).astype("float32")

    x_corrupted = x.copy()

    idx = np.random.permutation(N)
    anom_idx = idx[:num_anom]
    normal_idx = idx[num_anom:]

    n_swap = int(feature_swap_ratio * num_anom)
    n_struct_only = int(structure_only_ratio * num_anom)
    n_noise = num_anom - n_swap - n_struct_only

    swap_nodes = anom_idx[:n_swap]
    noise_nodes = anom_idx[n_swap:n_swap + n_noise]
    struct_nodes = anom_idx[n_swap + n_noise:]

    # ---------- Feature-swap anomalies ----------
    for i in swap_nodes:
        donor = np.random.choice(normal_idx)
        alpha = np.random.uniform(0.08, 0.20)
        x_corrupted[i] = alpha * x[i] + (1.0 - alpha) * x[donor]

    # ---------- Gaussian feature anomalies ----------
    if len(noise_nodes) > 0:
        noise = feat_noise_scale * np.random.randn(len(noise_nodes), F).astype("float32")
        x_corrupted[noise_nodes] += noise

    x_corrupted = np.clip(
        x_corrupted,
        np.percentile(x, 1) - 0.5,
        np.percentile(x, 99) + 0.5
    )

    # ---------- Mild structure perturbation ----------
    structural_targets = np.concatenate([swap_nodes, struct_nodes], axis=0) if len(struct_nodes) > 0 else swap_nodes

    for i in structural_targets:
        neighbors = np.where(A[i] > 0)[0]
        neighbors = neighbors[neighbors != i]
        deg = len(neighbors)
        if deg <= 1:
            continue

        num_rewire = max(1, int(edge_perturb_ratio * deg))
        num_rewire = min(num_rewire, max(1, deg // 3))

        remove_set = np.random.choice(neighbors, size=num_rewire, replace=False)

        A[i, remove_set] = 0.0
        A[remove_set, i] = 0.0

        existing_after = set(np.where(A[i] > 0)[0].tolist())
        forbidden = existing_after.union({i})

        deg_all = A.sum(axis=1)
        deg_i = deg_all[i]
        candidates = np.array(
            [n for n in range(N) if n not in forbidden and abs(deg_all[n] - deg_i) <= 3],
            dtype=np.int32
        )

        if len(candidates) < num_rewire:
            candidates = np.array([n for n in range(N) if n not in forbidden], dtype=np.int32)

        if len(candidates) == 0:
            continue

        add_size = min(num_rewire, len(candidates))
        add_set = np.random.choice(candidates, size=add_size, replace=False)

        A[i, add_set] = 1.0
        A[add_set, i] = 1.0

    labels = np.zeros(N, dtype=np.int32)
    labels[anom_idx] = 1

    A_sp_new = sparse.csr_matrix(A)
    return x_corrupted.astype("float32"), A_sp_new, labels, anom_idx, normal_idx


# ============================================================
# 6. Loss helpers
# ============================================================

def huber_elementwise(y_true, y_pred, delta=1.0):
    error = y_true - y_pred
    abs_error = tf.abs(error)
    quadratic = tf.minimum(abs_error, delta)
    loss = 0.5 * tf.square(quadratic) + delta * (abs_error - quadratic)
    return loss


def zscore_normalize(v, eps=1e-6):
    v = tf.convert_to_tensor(v, dtype=tf.float32)
    return (v - tf.reduce_mean(v)) / (tf.math.reduce_std(v) + eps)


# ============================================================
# 7. Encoder
# ============================================================

class GraphLowHighEncoder(Model):
    def __init__(self, hidden_dim: int, out_dim: int, dropout_rate: float = 0.04, **kwargs):
        super().__init__(**kwargs)
        self.low_lin = layers.Dense(hidden_dim, use_bias=True)
        self.high_lin = layers.Dense(hidden_dim, use_bias=True)
        self.out_lin = layers.Dense(out_dim, use_bias=True)
        self.gate_mlp = tf.keras.Sequential([
            layers.Dense(hidden_dim, activation="relu"),
            layers.Dense(1, activation="sigmoid"),
        ])
        self.dropout = layers.Dropout(dropout_rate)

    def sparse_mm(self, a_sp, x):
        return tf.sparse.sparse_dense_matmul(a_sp, x)

    def call(self, inputs, training=False):
        x, a_norm = inputs
        x = self.dropout(x, training=training)

        x_low = self.low_lin(x)
        h_low = self.sparse_mm(a_norm, x_low)
        h_low = tf.nn.relu(h_low)

        x_high = self.high_lin(x)
        ax_high = self.sparse_mm(a_norm, x_high)
        h_high = x_high - ax_high
        h_high = tf.nn.relu(h_high)

        gate_input = tf.concat([h_low, h_high], axis=-1)
        gate = self.gate_mlp(gate_input)

        h = gate * h_low + (1.0 - gate) * h_high
        z = self.out_lin(h)
        return z, gate, h_low, h_high


# ============================================================
# 8. Decoders and statistics heads
# ============================================================

class AttributeDecoder(Model):
    def __init__(self, emb_dim: int, out_dim: int, **kwargs):
        super().__init__(**kwargs)
        self.mlp = tf.keras.Sequential([
            layers.Dense(emb_dim, activation="relu"),
            layers.Dense(out_dim, activation=None),
        ])

    def call(self, z, training=False):
        return self.mlp(z)


class StructureDecoder(Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def edge_logits(self, z, edge_index):
        src = edge_index[0]
        dst = edge_index[1]
        z_src = tf.gather(z, src)
        z_dst = tf.gather(z, dst)
        logits = tf.reduce_sum(z_src * z_dst, axis=-1)
        return logits

    def call(self, z, edge_index_pos, edge_index_neg):
        pos_logits = self.edge_logits(z, edge_index_pos)
        neg_logits = self.edge_logits(z, edge_index_neg)
        logits = tf.concat([pos_logits, neg_logits], axis=0)
        labels = tf.concat([
            tf.ones_like(pos_logits, dtype=tf.float32),
            tf.zeros_like(neg_logits, dtype=tf.float32)
        ], axis=0)
        return logits, labels, pos_logits, neg_logits


class FullCovarianceHead(Model):
    def __init__(self, emb_dim: int, out_dim: int, eps=1e-4, **kwargs):
        super().__init__(**kwargs)
        self.out_dim = out_dim
        self.eps = eps
        self.num_tril = out_dim * (out_dim + 1) // 2
        self.proj = layers.Dense(self.num_tril, activation=None)

        idx = np.tril_indices(out_dim)
        self.row_idx = tf.constant(idx[0], dtype=tf.int32)
        self.col_idx = tf.constant(idx[1], dtype=tf.int32)

    def call(self, z, training=False):
        vec = self.proj(z)
        D = self.out_dim

        def build_single(v):
            L = tf.scatter_nd(
                indices=tf.stack([self.row_idx, self.col_idx], axis=1),
                updates=v,
                shape=(D, D)
            )
            diag = tf.linalg.diag_part(L)
            diag = tf.nn.softplus(diag) + self.eps
            L = tf.linalg.set_diag(L, diag)
            Sigma = tf.matmul(L, L, transpose_b=True) + self.eps * tf.eye(D, dtype=tf.float32)
            return Sigma

        Sigma_hat = tf.map_fn(build_single, vec, fn_output_signature=tf.float32)
        return Sigma_hat


# ============================================================
# 10. LACE-DR
# ============================================================

class LACE_DR(Model):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 24,
        out_dim: int = 24,
        lambda_attr: float = 0.65,
        lambda_struct: float = 0.65,
        lambda_con: float = 1.25,
        lambda_stats: float = 0.05,
        lambda_robust: float = 0.10,
        lambda_alpha: float = 0.01,
        tau_sim: float = 0.42,
        tau_mahal: float = 0.85,
        robust_radius: float = 2.0,
        shrinkage: float = 0.14,
        k_hop: int = 2,
        dropout_rate: float = 0.04,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.encoder = GraphLowHighEncoder(hidden_dim=hidden_dim, out_dim=out_dim, dropout_rate=dropout_rate)
        self.attr_decoder = AttributeDecoder(out_dim, in_dim)
        self.struct_decoder = StructureDecoder()

        self.alpha_mlp = tf.keras.Sequential([
            layers.Dense(hidden_dim, activation="relu"),
            layers.Dense(1, activation="sigmoid"),
        ])

        self.mu_head = layers.Dense(out_dim, activation=None)
        self.cov_head = FullCovarianceHead(emb_dim=out_dim, out_dim=out_dim)

        self.stats_module = None
        self.k_hop = k_hop
        self.shrinkage = shrinkage

        self.lambda_attr = lambda_attr
        self.lambda_struct = lambda_struct
        self.lambda_con = lambda_con
        self.lambda_stats = lambda_stats
        self.lambda_robust = lambda_robust
        self.lambda_alpha = lambda_alpha

        self.tau_sim = tau_sim
        self.tau_mahal = tau_mahal
        self.robust_radius = robust_radius

    def call(self, inputs, training=False):
        x, a_norm = inputs
        z, gate, h_low, h_high = self.encoder([x, a_norm], training=training)
        return z
