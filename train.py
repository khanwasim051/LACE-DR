"""Train and evaluate LACE-DR on one anomaly-injected graph.

Protocol (as in the paper): anomalies are injected with a fixed seed, the model is trained
for a fixed number of epochs without using the labels, and the model after the final epoch
is evaluated (AUC-ROC and average precision). The injected labels are used only for evaluation.

Examples
--------
python train.py --dataset cora --seed 42
python train.py --dataset pubmed --seed 43 --noise 0.4
python train.py --dataset cora --seed 42 --ablation global_stats
"""
import argparse
import json
import os
import time

import numpy as np
import scipy.sparse as sp
import tensorflow as tf
from sklearn.metrics import roc_auc_score, average_precision_score

from lacedr import core as L
from lacedr.data import load_dataset, gcn_filter
from lacedr.model import LACE_DR_Vec, GraphCache, to_tf_sparse

ABLATIONS = ["global_stats", "no_con", "no_mahal_w", "no_robust", "low_only", "no_gate"]


def run(dataset, seed, noise=0.62, edge=0.20, ratio=0.05, swap=0.10, struct=0.08,
        epochs=100, lr=8e-4, ablation=None, verbose=True):
    L.set_seed(seed)
    x, a = load_dataset(dataset)
    x_c, a_c, y, _, _ = L.inject_feature_and_structural_anomalies(
        x, a, ratio=ratio, feat_noise_scale=noise, edge_perturb_ratio=edge,
        feature_swap_ratio=swap, structure_only_ratio=struct)
    A = sp.csr_matrix(a_c)
    A = ((A + A.T) > 0).astype(np.float32)

    t0 = time.time()
    g = GraphCache(A, k=2)
    A_tf = to_tf_sparse(sp.csr_matrix(gcn_filter(A)))
    fixed_neg = g.sample_neg_edges(g.pos_edges.shape[1], np.random.RandomState(seed + 12345))
    t_pre = time.time() - t0

    xt = tf.constant(x_c)
    m = LACE_DR_Vec(in_dim=x_c.shape[1], hidden_dim=20, out_dim=20,
                    lambda_attr=0.65, lambda_struct=0.65, lambda_con=1.25, lambda_stats=0.05,
                    lambda_robust=0.10, lambda_alpha=0.01, tau_sim=0.42, tau_mahal=0.85,
                    robust_radius=2.0, shrinkage=0.14, k_hop=2, dropout_rate=0.04)
    m.rng = np.random.RandomState(seed)
    m.abl = set() if not ablation else {ablation}
    opt = tf.keras.optimizers.Adam(lr)

    t0 = time.time()
    for ep in range(epochs):
        with tf.GradientTape() as tape:
            loss = m.compute_losses(xt, A_tf, g)["loss"]
        grads = tape.gradient(loss, m.trainable_variables)
        opt.apply_gradients([(gr, v) for gr, v in zip(grads, m.trainable_variables) if gr is not None])
        if verbose and (ep + 1) % 20 == 0:
            print(f"epoch {ep + 1:4d}  loss {float(loss):.4f}", flush=True)
    t_train = time.time() - t0

    # anomaly score: z-scored attribute error (w.r.t. observed attributes), structural error and
    # local Mahalanobis deviation, with weights selected on validation seeds 100-104
    t0 = time.time()
    s = m.anomaly_scores(xt, A_tf, g, fixed_neg, w_attr=0.6, w_struct=0.2, w_mahal=0.2, score_raw=True)
    t_inf = time.time() - t0
    return dict(dataset=dataset, seed=seed, noise=noise, edge=edge, ratio=ratio, swap=swap,
                struct=struct, ablation=ablation or "full", n_nodes=int(x_c.shape[0]),
                n_anomalies=int(y.sum()), auc=float(roc_auc_score(y, s)),
                ap=float(average_precision_score(y, s)), preprocess_s=t_pre,
                train_s=t_train, inference_s=t_inf)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="cora", choices=["cora", "citeseer", "pubmed", "blogcatalog"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--noise", type=float, default=0.62, help="std of the Gaussian attribute noise")
    p.add_argument("--edge", type=float, default=0.20, help="edge-perturbation strength")
    p.add_argument("--ratio", type=float, default=0.05, help="anomaly ratio")
    p.add_argument("--swap", type=float, default=0.10, help="share of feature-interpolation anomalies")
    p.add_argument("--struct", type=float, default=0.08, help="share of structure-only anomalies")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--ablation", default=None, choices=ABLATIONS)
    p.add_argument("--out", default=None, help="optional path of a JSON file for the result")
    a = p.parse_args()
    r = run(a.dataset, a.seed, a.noise, a.edge, a.ratio, a.swap, a.struct, a.epochs, ablation=a.ablation)
    print(f"{r['dataset']} seed {r['seed']}: AUC {r['auc']:.4f}  AP {r['ap']:.4f}  "
          f"(train {r['train_s']:.1f} s, inference {r['inference_s']:.2f} s)")
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump(r, open(a.out, "w"), indent=1)
