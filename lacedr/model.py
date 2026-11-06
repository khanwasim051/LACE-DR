"""
Vectorized LACE-DR.

Computes the same quantities as the original per-node implementation, but replaces the per-node Python loops with batched / sparse tensor operations:

- local mean, covariance (with trace-scaled shrinkage), inverse and Mahalanobis
  distance: sparse 2-hop indicator matrix R times per-node outer products,
  followed by one batched d x d inverse
- neighbourhood feature mean mu^x: R @ X
- distribution-aware contrastive loss: positives/negatives sampled once per epoch
  as padded index tensors, Mahalanobis weights by batched einsum
- full-covariance head: one matmul that scatters lower-triangular entries,
  instead of tf.map_fn over nodes
- positive edges precomputed once; negative edges sampled sparsely (uniform over
  non-edges) instead of enumerating all N^2/2 node pairs every epoch
- node-level structural error aggregation with np.add.at instead of a Python loop
"""
import numpy as np
import scipy.sparse as sp
import tensorflow as tf
from tensorflow.keras import layers, Model

from . import core as L


# ------------------------------------------------------------------
# Graph preprocessing (done once)
# ------------------------------------------------------------------

def khop_indicator(A, k=2, include_self=True):
    A = sp.csr_matrix(A > 0).astype(np.float32)
    N = A.shape[0]
    R = A.copy()
    P = A.copy()
    for _ in range(1, k):
        P = ((P @ A) > 0).astype(np.float32)
        R = ((R + P) > 0).astype(np.float32)
    if include_self:
        R = ((R + sp.eye(N, dtype=np.float32)) > 0).astype(np.float32)
    R = sp.csr_matrix(R)
    R.sort_indices()
    return R


def to_tf_sparse(M):
    M = sp.coo_matrix(M)
    t = tf.SparseTensor(np.vstack([M.row, M.col]).T.astype(np.int64),
                        M.data.astype(np.float32), M.shape)
    return tf.sparse.reorder(t)


class GraphCache:
    """Everything that depends only on the (fixed) graph."""

    def __init__(self, A, k=2):
        A = sp.csr_matrix(A)
        A.setdiag(0)
        A.eliminate_zeros()
        A = ((A + A.T) > 0).astype(np.float32)
        self.N = A.shape[0]
        self.R = khop_indicator(A, k=k, include_self=True)
        self.R_tf = to_tf_sparse(self.R)
        # dense route when 2-hop neighbourhoods are large (dense graphs): avoids an nnz(R) x d^2 gradient buffer
        self.dense = self.R.nnz > 0.02 * self.N * self.N
        self.R_d = tf.constant(self.R.toarray()) if self.dense else None
        self.K = np.asarray(self.R.sum(1)).ravel().astype(np.float32)          # |N_k(i)| (incl. self)
        self.K_tf = tf.constant(self.K[:, None])
        self.R_keys = (self.R.tocoo().row.astype(np.int64) * self.N + self.R.tocoo().col)
        self.R_keys.sort()
        coo = sp.triu(A, k=1).tocoo()
        self.pos_edges = np.vstack([coo.row, coo.col]).astype(np.int32)    # undirected, src < dst
        self.edge_keys = np.sort(coo.row.astype(np.int64) * self.N + coo.col)
        # off-diagonal neighbourhood pairs, for positive sampling
        Rc = self.R.tocoo()
        off = Rc.row != Rc.col
        self.pr, self.pc = Rc.row[off].astype(np.int64), Rc.col[off].astype(np.int64)
        # row boundaries of the (row-sorted) pair list, and row-aligned chunks of <= 1M pairs
        self.pr_start = np.searchsorted(self.pr, np.arange(self.N + 1))
        self.chunks, a = [], 0
        while a < self.N:
            b = max(a + 1, int(np.searchsorted(self.pr_start, self.pr_start[a] + 1_000_000, side="right")) - 1)
            b = min(b, self.N); self.chunks.append((a, b)); a = b

    def rmm(self, M):
        """R @ M (sum over each node's 2-hop neighbourhood)."""
        return tf.matmul(self.R_d, M) if self.dense else tf.sparse.sparse_dense_matmul(self.R_tf, M)

    # ---- sampling -------------------------------------------------
    def sample_neg_edges(self, num_neg, rng):
        N, out = self.N, np.empty(0, dtype=np.int64)
        while out.size < num_neg:
            m = int(1.3 * (num_neg - out.size)) + 16
            i = rng.randint(0, N, m).astype(np.int64)
            j = rng.randint(0, N, m).astype(np.int64)
            keep = i != j
            a, b = np.minimum(i, j)[keep], np.maximum(i, j)[keep]
            key = a * N + b
            pos = np.searchsorted(self.edge_keys, key)
            pos = np.minimum(pos, max(len(self.edge_keys) - 1, 0))
            is_edge = self.edge_keys[pos] == key if len(self.edge_keys) else np.zeros_like(key, bool)
            out = np.unique(np.concatenate([out, key[~is_edge]]))
        out = rng.permutation(out)[:num_neg]
        return np.vstack([out // N, out % N]).astype(np.int32)

    def sample_contrastive(self, rng, max_pos=10, num_neg=10):
        N = self.N
        # positives: up to max_pos random neighbours (without replacement), excluding self
        # processed in row-aligned chunks (same random stream and result as one global lexsort, far less memory)
        P = np.zeros((N, max_pos), np.int32)
        Pm = np.zeros((N, max_pos), np.float32)
        for a, b in self.chunks:
            lo, hi = self.pr_start[a], self.pr_start[b]
            sub = self.pr[lo:hi]
            order = np.lexsort((rng.rand(hi - lo), sub))
            rows, cols = sub[order], self.pc[lo:hi][order]
            rank = np.arange(hi - lo) - (self.pr_start[rows] - lo)
            keep = rank < max_pos
            P[rows[keep], rank[keep]] = cols[keep]
            Pm[rows[keep], rank[keep]] = 1.0
        # negatives: num_neg random nodes outside N_k(i) (without replacement)
        C = 4 * num_neg
        cand = rng.randint(0, N, (N, C)).astype(np.int64)
        key = np.arange(N)[:, None] * N + cand
        pos = np.minimum(np.searchsorted(self.R_keys, key), len(self.R_keys) - 1)
        valid = self.R_keys[pos] != key
        o = np.argsort(cand, axis=1, kind="stable")
        s = np.take_along_axis(cand, o, axis=1)
        dup_sorted = np.zeros_like(valid)
        dup_sorted[:, 1:] = s[:, 1:] == s[:, :-1]
        dup = np.zeros_like(valid)
        np.put_along_axis(dup, o, dup_sorted, axis=1)
        valid &= ~dup
        take = valid & (np.cumsum(valid, axis=1) <= num_neg)
        Q = np.zeros((N, num_neg), np.int32)
        Qm = np.zeros((N, num_neg), np.float32)
        r, c = np.nonzero(take)
        slot = np.cumsum(take, axis=1)[r, c] - 1
        Q[r, slot] = cand[r, c]
        Qm[r, slot] = 1.0
        return P, Pm, Q, Qm


# ------------------------------------------------------------------
# Batched local elliptical statistics
# ------------------------------------------------------------------

def local_stats(z, g, shrinkage, eps=1e-4):
    N, D = tf.shape(z)[0], z.shape[1]
    K = g.K_tf
    mu = g.rmm(z) / K                               # (N, D)
    zz = tf.reshape(z[:, :, None] * z[:, None, :], (-1, D * D))
    Szz = tf.reshape(g.rmm(zz), (-1, D, D))         # sum_j z_j z_j^T
    S = (Szz - K[:, :, None] * mu[:, :, None] * mu[:, None, :]) / tf.maximum(K - 1.0, 1.0)[:, :, None]
    I = tf.eye(D, dtype=tf.float32)
    tr = tf.linalg.trace(S) / float(D)
    Sigma = (1.0 - shrinkage) * S + shrinkage * tr[:, None, None] * I + eps * I
    Sigma_inv = tf.linalg.inv(Sigma)
    diff = z - mu
    d_self = tf.einsum("nd,nde,ne->n", diff, Sigma_inv, diff)
    return mu, Sigma, Sigma_inv, d_self


class FullCovarianceHeadVec(Model):
    """Same parameterisation as FullCovarianceHead, built without map_fn."""

    def __init__(self, out_dim, eps=1e-4, **kw):
        super().__init__(**kw)
        self.D, self.eps = out_dim, eps
        T = out_dim * (out_dim + 1) // 2
        self.proj = layers.Dense(T, activation=None)
        r, c = np.tril_indices(out_dim)
        P = np.zeros((T, out_dim * out_dim), np.float32)
        P[np.arange(T), r * out_dim + c] = 1.0
        self.scatter = tf.constant(P)

    def call(self, z, training=False):
        D = self.D
        Lm = tf.reshape(self.proj(z) @ self.scatter, (-1, D, D))
        diag = tf.nn.softplus(tf.linalg.diag_part(Lm)) + self.eps
        Lm = tf.linalg.set_diag(Lm, diag)
        return tf.matmul(Lm, Lm, transpose_b=True) + self.eps * tf.eye(D)


# ------------------------------------------------------------------
# Model
# ------------------------------------------------------------------

class LACE_DR_Vec(L.LACE_DR):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.cov_head = FullCovarianceHeadVec(kw.get("out_dim", 32))
        self.rng = np.random.RandomState(0)
        self.abl = set()   # ablation switches: global_stats, no_mahal_w, no_gate, no_robust, no_con, low_only

    def stats(self, z, g):
        if "global_stats" not in self.abl:
            return local_stats(z, g, self.shrinkage)
        N = tf.cast(tf.shape(z)[0], tf.float32); D = z.shape[1]
        mu0 = tf.reduce_mean(z, axis=0, keepdims=True)
        S = tf.matmul(z - mu0, z - mu0, transpose_a=True) / tf.maximum(N - 1.0, 1.0)
        I = tf.eye(D, dtype=tf.float32)
        Sig = (1.0 - self.shrinkage) * S + self.shrinkage * tf.linalg.trace(S) / float(D) * I + 1e-4 * I
        Sinv = tf.linalg.inv(Sig); diff = z - mu0
        d_self = tf.einsum("nd,de,ne->n", diff, Sinv, diff)
        n = tf.shape(z)[0]
        return (tf.tile(mu0, [n, 1]), tf.tile(Sig[None], [n, 1, 1]), tf.tile(Sinv[None], [n, 1, 1]), d_self)

    def encode(self, x, a_norm, training):
        if "low_only" in self.abl and not getattr(self, "_low_patched", False):
            enc = self.encoder
            enc.__dict__["gate_mlp"] = lambda t: tf.ones_like(t[:, :1])
            self._low_patched = True
        return self.encoder([x, a_norm], training=training)

    def x_mu(self, x, g):
        """Neighbourhood mean of the input features (constant: computed once and cached)."""
        if getattr(self, "_xmu_key", None) != id(x):
            self._xmu = tf.stop_gradient(tf.sparse.sparse_dense_matmul(g.R_tf, x) / g.K_tf)
            self._xmu_key = id(x)
        return self._xmu

    def contrastive(self, z, mu, Sigma_inv, g):
        P, Pm, Q, Qm = g.sample_contrastive(self.rng)
        zn = tf.nn.l2_normalize(z, axis=1)
        def part(idx, m):
            zj = tf.gather(z, idx)                                       # (N, k, D)
            sim = tf.einsum("nkd,nd->nk", tf.gather(zn, idx), zn)
            diff = zj - mu[:, None, :]
            d = tf.einsum("nkd,nde,nke->nk", diff, Sigma_inv, diff)
            if "no_mahal_w" in self.abl: d = tf.zeros_like(d)
            return tf.reduce_sum(m * tf.exp(-d / self.tau_mahal) * tf.exp(sim / self.tau_sim), axis=1)
        num = part(P, Pm)
        den = num + part(Q, Qm) + 1e-8
        li = -tf.math.log(num / den + 1e-8)
        use = tf.constant(((Pm.sum(1) > 0) & (Qm.sum(1) > 0)).astype(np.float32))
        return tf.reduce_sum(li * use) / tf.maximum(tf.reduce_sum(use), 1.0)

    def compute_losses(self, x, a_norm, g):
        z, gate, _, _ = self.encode(x, a_norm, True)
        mu_z, cov_z, cov_inv_z, d_self = self.stats(z, g)
        x_mu = self.x_mu(x, g)
        alpha = tf.ones_like(x[:, :1]) if "no_gate" in self.abl else self.alpha_mlp(z)
        x_tilde = alpha * x + (1.0 - alpha) * x_mu
        x_hat = self.attr_decoder(z, training=True)
        attr_err_vec = tf.reduce_mean(L.huber_elementwise(x_tilde, x_hat, delta=1.0), axis=1)
        L_attr = tf.reduce_mean(attr_err_vec)

        neg = g.sample_neg_edges(g.pos_edges.shape[1], self.rng)
        logits, labels, _, _ = self.struct_decoder(z, tf.constant(g.pos_edges), tf.constant(neg))
        L_struct = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits))

        L_con = tf.constant(0.0) if "no_con" in self.abl else self.contrastive(z, mu_z, cov_inv_z, g)

        mu_hat, cov_hat = self.mu_head(z), self.cov_head(z)
        L_stats = (tf.reduce_mean(L.huber_elementwise(mu_z, mu_hat, delta=1.0))
                   + tf.reduce_mean(tf.norm(cov_z - cov_hat, ord="fro", axis=[-2, -1])))
        L_robust = tf.constant(0.0) if "no_robust" in self.abl else tf.reduce_mean(tf.square(tf.nn.relu(d_self - self.robust_radius)))
        L_alpha = tf.reduce_mean(tf.exp(-d_self)[:, None] * tf.square(alpha - 1.0))

        total = (self.lambda_attr * L_attr + self.lambda_struct * L_struct + self.lambda_con * L_con
                 + self.lambda_stats * L_stats + self.lambda_robust * L_robust + self.lambda_alpha * L_alpha)
        return dict(loss=total, L_attr=L_attr, L_struct=L_struct, L_con=L_con, L_stats=L_stats,
                    L_robust=L_robust, L_alpha=L_alpha, z=z, d_self=d_self)

    def anomaly_scores(self, x, a_norm, g, neg_edges_np, w_attr=0.6, w_struct=0.2, w_mahal=0.2, score_raw=True):
        z, _, _, _ = self.encode(x, a_norm, False)
        _, _, _, d_self = self.stats(z, g)
        x_mu = self.x_mu(x, g)
        alpha = tf.ones_like(x[:, :1]) if "no_gate" in self.abl else self.alpha_mlp(z)
        x_tilde = alpha * x + (1.0 - alpha) * x_mu
        attr_err = tf.reduce_mean(L.huber_elementwise(x if score_raw else x_tilde, self.attr_decoder(z, training=False), delta=1.0), axis=1)
        pe, ne = g.pos_edges, neg_edges_np
        pl = self.struct_decoder.edge_logits(z, tf.constant(pe))
        nl = self.struct_decoder.edge_logits(z, tf.constant(ne))
        perr = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(pl), logits=pl).numpy()
        nerr = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(nl), logits=nl).numpy()
        err = np.zeros(g.N, np.float32)
        deg = np.zeros(g.N, np.float32)
        for E, e in ((pe, perr), (ne, nerr)):
            for side in (0, 1):
                np.add.at(err, E[side], e)
                np.add.at(deg, E[side], 1.0)
        s_err = tf.constant(err / (deg + 1e-6))
        return (w_attr * L.zscore_normalize(attr_err) + w_struct * L.zscore_normalize(s_err)
                + w_mahal * L.zscore_normalize(d_self)).numpy()
