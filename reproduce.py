"""Reproduce the LACE-DR results reported in the paper.

python reproduce.py --group main         # 4 datasets x 5 seeds (main results)
python reproduce.py --group robustness   # Cora: anomaly types, noise levels, edge perturbation, anomaly ratio
python reproduce.py --group ablation     # ablation variants on Cora, Citeseer and BlogCatalog
python reproduce.py --group all

Each run is stored in results/<name>_s<seed>.json and skipped if it already exists, so the
script can be stopped and restarted. A summary (mean and 95% t-confidence interval) is printed
at the end.
"""
import argparse
import glob
import json
import os
import subprocess
import sys

import numpy as np
from scipy import stats

S5, S3 = [42, 43, 44, 45, 46], [42, 43, 44]
D = dict(noise=0.62, edge=0.20, ratio=0.05, swap=0.10, struct=0.08)


def cfg(**kw):
    c = dict(D); c.update(kw); return c


GROUPS = {
    "main": [(f"{ds}_default", ds, S5, cfg(), None) for ds in ["cora", "citeseer", "pubmed", "blogcatalog"]],
    "robustness":
        [("cora_type_attr", "cora", S5, cfg(swap=0.0, struct=0.0), None),
         ("cora_type_struct", "cora", S5, cfg(swap=0.0, struct=1.0), None),
         ("cora_type_mixed", "cora", S5, cfg(swap=1.0, struct=0.0), None)]
        + [(f"cora_noise_{v}", "cora", S3, cfg(noise=v), None) for v in [0.2, 0.4, 1.0]]
        + [(f"cora_edge_{v}", "cora", S3, cfg(edge=v), None) for v in [0.1, 0.3]]
        + [(f"cora_ratio_{v}", "cora", S3, cfg(ratio=v), None) for v in [0.02, 0.10]],
    "ablation":
        [(f"{ds}_abl_{v}", ds, S5, cfg(), v) for ds in ["cora", "citeseer"]
         for v in ["global_stats", "no_con", "no_mahal_w", "no_robust", "low_only", "no_gate"]]
        + [(f"blogcatalog_abl_{v}", "blogcatalog", S5, cfg(), v) for v in ["global_stats", "low_only", "no_gate"]],
}


def ci(v):
    v = np.asarray(v, float)
    if len(v) < 2:
        return v.mean(), float("nan")
    return v.mean(), stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--group", default="main", choices=list(GROUPS) + ["all"])
    p.add_argument("--results", default="results")
    a = p.parse_args()
    jobs = sum(GROUPS.values(), []) if a.group == "all" else GROUPS[a.group]
    here = os.path.dirname(os.path.abspath(__file__))
    for name, ds, seeds, c, abl in jobs:
        for s in seeds:
            out = os.path.join(a.results, f"{name}_s{s}.json")
            if os.path.exists(out):
                continue
            cmd = [sys.executable, os.path.join(here, "train.py"), "--dataset", ds, "--seed", str(s), "--out", out]
            for k, v in c.items():
                cmd += [f"--{k}", str(v)]
            if abl:
                cmd += ["--ablation", abl]
            print(">>", name, "seed", s, flush=True)
            subprocess.run(cmd, check=True)
    print("\nSummary (mean +- half-width of the 95% confidence interval)")
    for name, ds, seeds, c, abl in jobs:
        rs = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(a.results, f"{name}_s*.json")))]
        if not rs:
            continue
        auc, ap = ci([r["auc"] for r in rs]), ci([r["ap"] for r in rs])
        print(f"{name:28s} n={len(rs)}  AUC {auc[0]:.3f} +- {auc[1]:.3f}   AP {ap[0]:.3f} +- {ap[1]:.3f}")


if __name__ == "__main__":
    main()
