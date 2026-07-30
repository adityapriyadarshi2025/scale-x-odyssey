"""
anomaly.py — anomaly / out-of-distribution detection (Phase 4 bonus).

The classifier always emits one of the 5 classes, even for an image that is none
of them. This adds a detector that flags "this doesn't look like any of my 5
classes" from the model's own outputs — no retraining, no extra labels. Two
standard post-hoc scores:

  - MSP    : max softmax probability      (high = confident / in-distribution)
  - energy : logsumexp of the logits      (high = in-distribution; Liu et al. 2020)

We *quantify* the detector by reusing the real-Hubble OOD set as the "anomalies":
how well does each score separate in-distribution test images from OOD ones
(AUROC), and — at a threshold calibrated to a 5 % false-positive rate on
in-distribution data — what fraction of OOD images are caught. This directly
operationalises the §5 OOD finding into a working detector.

Run from repo root:
    python -m src.anomaly --data_root /content/drive/MyDrive/scale-odyssey/data \
                          --ood /content/drive/MyDrive/scale-odyssey/data/ood
"""

import os
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader

from . import data as D
from . import transforms as T
from . import eval as E
from .data import LABELS


@torch.no_grad()
def score_records(model, records, cfg, device):
    """Return (msp, energy) arrays — one score per image."""
    d = cfg["data"]
    tf = T.eval_transforms(d["image_size"], d["norm_mean"], d["norm_std"])
    ld = DataLoader(D.AstroDataset(records, tf), batch_size=cfg["train"]["batch_size"],
                    shuffle=False, num_workers=cfg["train"]["num_workers"])
    msp, energy = [], []
    for x, _ in ld:
        logits = model(x.to(device))
        msp.append(torch.softmax(logits, 1).max(1).values.cpu().numpy())
        energy.append(torch.logsumexp(logits, 1).cpu().numpy())
    return np.concatenate(msp), np.concatenate(energy)


def _ood_records(ood_dir):
    if not os.path.isdir(ood_dir):
        ood_dir = "/content/drive/MyDrive/scale-odyssey/data/ood"
    return [{"path": os.path.join(ood_dir, c, f), "label": LABELS[c],
             "class": c, "origin": f, "source": "ood"}
            for c in os.listdir(ood_dir) if os.path.isdir(os.path.join(ood_dir, c))
            for f in os.listdir(os.path.join(ood_dir, c)) if f.lower().endswith(".jpg")]


def main(config, ckpt, data_root, ood_dir):
    from sklearn.metrics import roc_auc_score
    cfg = E.load_config(config)
    if data_root:
        cfg["paths"]["data_root"] = data_root
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = ckpt or os.path.join(cfg["paths"]["artifacts_dir"], "best_model.pth")
    model = E.load_model(cfg, ckpt, device)

    id_recs = E.build_splits(cfg)["test"]
    ood_recs = _ood_records(ood_dir)
    print(f"in-distribution test: {len(id_recs)}   OOD: {len(ood_recs)}")

    id_msp, id_en = score_records(model, id_recs, cfg, device)
    ood_msp, ood_en = score_records(model, ood_recs, cfg, device)

    print("\n=== anomaly / OOD detection (OOD = the anomaly class) ===")
    for name, id_s, ood_s in [("MSP", id_msp, ood_msp), ("energy", id_en, ood_en)]:
        # in-distribution scores are HIGH -> anomaly-ness = -score, OOD labelled 1
        y = np.r_[np.zeros(len(id_s)), np.ones(len(ood_s))]
        s = np.r_[-id_s, -ood_s]
        auroc = roc_auc_score(y, s)
        thr = float(np.percentile(id_s, 5))          # 5% false-positive rate on ID
        caught = float((ood_s < thr).mean())
        print(f"  {name:7} AUROC={auroc:.3f}   @5%FPR flags {caught*100:4.1f}% of OOD"
              f"   (threshold {thr:.3f})")
    print("\nAUROC > 0.5 means the score separates unfamiliar images from the 5 "
          "known classes; higher is better.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--data_root", default=None)
    ap.add_argument("--ood", default="/content/so_data/ood")
    a = ap.parse_args()
    main(a.config, a.checkpoint, a.data_root, a.ood)
