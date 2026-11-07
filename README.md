# LACE-DR

**Local Adaptive Contrastive Encoding with Distributionally-Robust Reconstruction for Anomaly Detection in Attributed Networks**

This repository contains the implementation of LACE-DR, an unsupervised node-anomaly detector for attributed graphs, and the scripts that reproduce the LACE-DR results reported in the paper (manuscript under review).

LACE-DR combines:

* a **bi-filter graph encoder** that fuses low-pass and high-pass graph signals through a learned gate;
* a **local elliptical reference model**: for every node, the mean and shrinkage covariance of the embeddings in its 2-hop neighbourhood, and the node's squared Mahalanobis deviation from it;
* a **distribution-aware contrastive objective** that weights positive and negative pairs by their Mahalanobis consistency;
* **distributionally robust dual reconstruction** of attributes (Huber loss, confidence-gated target), structure (observed edges and sampled non-edges) and neighbourhood statistics, with a robust deviation penalty and a confidence regulariser.

The anomaly score combines the z-scored attribute reconstruction error (with respect to the observed attributes), the structural reconstruction error and the local Mahalanobis deviation, with weights 0.6 / 0.2 / 0.2.

## Installation

Python 3.10–3.13.

```bash
pip install -r requirements.txt
```

## Data

Cora, Citeseer and PubMed are downloaded automatically on first use. BlogCatalog must be placed in `data/BlogCatalog.mat` (see `data/README.md`).

## Usage

Train and evaluate on one anomaly-injected graph:

```bash
python train.py --dataset cora --seed 42
# cora seed 42: AUC 0.9144  AP 0.8589
```

Options:

| Option | Meaning | Default |
|---|---|---|
| `--dataset` | `cora`, `citeseer`, `pubmed`, `blogcatalog` | `cora` |
| `--seed` | random seed (anomaly injection and training) | 42 |
| `--noise` | standard deviation of the Gaussian attribute noise | 0.62 |
| `--edge` | edge-perturbation strength | 0.20 |
| `--ratio` | anomaly ratio | 0.05 |
| `--swap`, `--struct` | shares of feature-interpolation and structure-only anomalies | 0.10, 0.08 |
| `--ablation` | `global_stats`, `no_con`, `no_mahal_w`, `no_robust`, `low_only`, `no_gate` | none |
| `--out` | write the result to a JSON file | none |

Reproduce the experiments of the paper:

```bash
python reproduce.py --group main         # 4 datasets x 5 seeds
python reproduce.py --group robustness   # Cora: anomaly types, noise levels, edge perturbation, anomaly ratio
python reproduce.py --group ablation     # ablation study
```

Finished runs are skipped, so the script can be interrupted and restarted. It prints the mean and the 95% confidence interval of AUC-ROC and AP for every setting.

## Evaluation protocol

* Anomalies are injected with a fixed seed (Gaussian attribute noise, feature interpolation with partial edge removal, and structure-only edge removal). The labels are used **only for evaluation**.
* The model is trained for a fixed 100 epochs (Adam, learning rate 8e-4) and the model **after the final epoch** is evaluated; there is no label-based model selection or early stopping.
* The weights of the anomaly score were selected on validation graphs generated with seeds 100–104, which are disjoint from the seeds used for the reported results (42–46).

## Expected results

LACE-DR, default anomaly mixture, mean over seeds 42–46 ± half-width of the 95% confidence interval:

| Dataset | AUC-ROC | AP |
|---|---|---|
| Cora | 0.915 ± 0.014 | 0.855 ± 0.008 |
| Citeseer | 0.922 ± 0.008 | 0.848 ± 0.003 |
| PubMed | 0.914 ± 0.006 | 0.849 ± 0.003 |
| BlogCatalog | 0.916 ± 0.005 | 0.853 ± 0.002 |

Robustness on Cora (AUC-ROC / AP):

| Setting | LACE-DR |
|---|---|
| Gaussian attribute anomalies only | 1.000 / 0.992 |
| Structure-only anomalies only | 0.498 / 0.054 |
| Interpolation + edge removal only | 0.246 / 0.033 |
| Noise σ = 0.2 / 0.4 / 1.0 | 0.900 / 0.832, 0.908 / 0.850, 0.908 / 0.855 |
| Edge perturbation 0.10 / 0.30 | 0.905 / 0.851, 0.907 / 0.853 |
| Anomaly ratio 0.02 / 0.10 | 0.914 / 0.846, 0.903 / 0.860 |

As discussed in the paper, LACE-DR does not detect purely structural anomalies or anomalies produced by feature interpolation with edge removal.

Small numerical differences can occur across hardware and TensorFlow versions. On Cora, we obtained identical results on a 2-core Intel Xeon CPU and on a Windows laptop.


## Repository structure

```
lacedr/core.py    anomaly injection, encoder, decoders, base model
lacedr/model.py   vectorised LACE-DR (local statistics, losses, anomaly score)
lacedr/data.py    dataset loading
train.py          single run
reproduce.py      experiments of the paper
```

## Citation

The paper is currently under review. Citation details will be added after publication.

## License

MIT (see `LICENSE`).
