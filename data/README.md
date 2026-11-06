# Data

* **Cora, Citeseer, PubMed** are downloaded automatically into `data/planetoid/` the first time they are used (Planetoid splits from https://github.com/kimiyoung/planetoid).
* **BlogCatalog** must be placed here as `data/BlogCatalog.mat`, a MATLAB file with the fields `Network` (adjacency matrix) and `Attributes` (node attributes). The version used in the paper has 5,196 nodes, 172,783 undirected edges and 8,189 attributes.

The anomalies are injected by the code (see `lacedr/core.py`, `inject_feature_and_structural_anomalies`), so no labels need to be downloaded.
