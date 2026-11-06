"""Dataset loading. Cora, Citeseer and PubMed are downloaded automatically (Planetoid splits);
BlogCatalog is read from data/BlogCatalog.mat."""
import os
import pickle
import urllib.request
import numpy as np
import scipy.sparse as sp

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
PLANETOID_URL = "https://raw.githubusercontent.com/kimiyoung/planetoid/master/data/ind.{name}.{part}"
PARTS = ["x", "tx", "allx", "graph", "test.index"]


def gcn_filter(A):
    """Symmetric normalisation D^{-1/2} (A + I) D^{-1/2}."""
    A = sp.csr_matrix(A) + sp.eye(A.shape[0])
    d = np.asarray(A.sum(1)).ravel()
    D = sp.diags(1 / np.sqrt(d))
    return (D @ A @ D).toarray() if A.shape[0] < 5000 else (D @ A @ D)


def _download_planetoid(name, folder):
    os.makedirs(folder, exist_ok=True)
    for part in PARTS:
        path = os.path.join(folder, f"ind.{name}.{part}")
        if not os.path.exists(path):
            print(f"downloading ind.{name}.{part}")
            urllib.request.urlretrieve(PLANETOID_URL.format(name=name, part=part), path)


def load_planetoid(name):
    folder = os.path.join(ROOT, "planetoid")
    _download_planetoid(name, folder)
    objs = []
    for n in ["x", "tx", "allx", "graph"]:
        with open(os.path.join(folder, f"ind.{name}.{n}"), "rb") as f:
            objs.append(pickle.load(f, encoding="latin1"))
    x, tx, allx, graph = objs
    test_idx = [int(l) for l in open(os.path.join(folder, f"ind.{name}.test.index"))]
    test_sorted = np.sort(test_idx)
    if name == "citeseer":  # isolated test nodes are missing from tx
        full = range(min(test_idx), max(test_idx) + 1)
        tx_ext = sp.lil_matrix((len(full), x.shape[1]))
        tx_ext[test_sorted - min(test_sorted), :] = tx
        tx = tx_ext
    feats = sp.vstack((allx, tx)).tolil()
    feats[test_idx, :] = feats[test_sorted, :]
    N = feats.shape[0]
    rows, cols = [], []
    for k, v in graph.items():
        for j in v:
            if k < N and j < N:
                rows.append(k); cols.append(j)
    A = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(N, N))
    A = ((A + A.T) > 0).astype(np.float32); A.setdiag(0); A.eliminate_zeros()
    return np.asarray(feats.todense(), dtype=np.float32), A


def load_mat(path):
    from scipy.io import loadmat
    d = loadmat(path)
    a = sp.csr_matrix(d["Network"]).astype(np.float32)
    x = d["Attributes"]
    x = x.toarray() if sp.issparse(x) else np.asarray(x)
    a = ((a + a.T) > 0).astype(np.float32); a.setdiag(0); a.eliminate_zeros()
    return x.astype(np.float32), a


def load_dataset(name):
    """Returns (attributes [N x F] float32, symmetric adjacency without self-loops, CSR)."""
    key = name.lower()
    if key in ("cora", "citeseer", "pubmed"):
        return load_planetoid(key)
    if key == "blogcatalog":
        path = os.path.join(ROOT, "BlogCatalog.mat")
        if not os.path.exists(path):
            raise FileNotFoundError("Place BlogCatalog.mat (fields 'Network' and 'Attributes') in the data/ folder.")
        return load_mat(path)
    raise ValueError(f"Unknown dataset: {name}")
