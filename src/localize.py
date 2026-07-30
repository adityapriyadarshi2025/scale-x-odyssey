"""
localize.py — weakly-supervised object localization via Grad-CAM (Phase 4 bonus).

Turns the classifier's Grad-CAM attention into a bounding box around the
celestial object — with NO bounding-box labels (weakly-supervised, the classic
CAM-localisation trick). For each image: Grad-CAM heatmap -> threshold at half
its peak -> largest connected region -> its bounding box. Produces a grid of
localised test images.

Run from repo root:
    python -m src.localize --checkpoint <best_model.pth> --data_root <data> --n 12
"""

import os
import argparse

import numpy as np
import torch
from PIL import Image

from . import transforms as T
from . import eval as E


def bbox_from_heat(heat, frac=0.5):
    """Bounding box of the largest region where heat >= frac * peak. Returns
    (x0, y0, x1, y1) in heatmap pixels, or None."""
    mask = heat >= frac * float(heat.max())
    try:
        from scipy import ndimage
        lbl, n = ndimage.label(mask)
        if n > 0:
            sizes = ndimage.sum(mask, lbl, range(1, n + 1))
            mask = lbl == (int(np.argmax(sizes)) + 1)   # keep largest component
    except Exception:
        pass
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def localize_one(model, cam, pil, cfg, device):
    d = cfg["data"]; size = d["image_size"]
    tf = T.eval_transforms(size, d["norm_mean"], d["norm_std"])
    x = tf(pil.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = int(model(x).argmax(1))
    heat = cam(input_tensor=x)[0]                       # size x size, in [0,1]
    box = bbox_from_heat(heat)
    rgb = np.array(pil.convert("RGB").resize((size, size)))
    return rgb, box, cfg["classes"][pred]


def main(config, ckpt, data_root, n):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from pytorch_grad_cam import GradCAM

    cfg = E.load_config(config)
    if data_root:
        cfg["paths"]["data_root"] = data_root
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = E.load_model(cfg, ckpt or os.path.join(cfg["paths"]["artifacts_dir"], "best_model.pth"), device)
    cam = GradCAM(model=model, target_layers=[E._target_layer(model)])

    # balanced sample: a few per class (compact objects show tight boxes; nebulae,
    # being diffuse, are the hard case — so show all classes, not just the first n)
    from collections import defaultdict
    by_cls = defaultdict(list)
    for r in E.build_splits(cfg)["test"]:
        by_cls[r["class"]].append(r)
    per = max(1, n // len(cfg["classes"]))
    recs = [r for c in cfg["classes"] for r in by_cls[c][:per]][:n]
    cols = 5
    rows = (len(recs) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    axes = np.array(axes).reshape(-1)
    for i, r in enumerate(recs):
        rgb, box, pred = localize_one(model, cam, Image.open(r["path"]), cfg, device)
        ax = axes[i]
        ax.imshow(rgb); ax.axis("off"); ax.set_title(pred, fontsize=8)
        if box:
            x0, y0, x1, y1 = box
            ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0,
                                   fill=False, edgecolor="lime", linewidth=2))
    for j in range(len(recs), len(axes)):
        axes[j].axis("off")
    fig.suptitle("Weakly-supervised localization (Grad-CAM bounding box)")
    fig.tight_layout()
    out = os.path.join(cfg["paths"]["artifacts_dir"], "localization.png")
    os.makedirs(cfg["paths"]["artifacts_dir"], exist_ok=True)
    fig.savefig(out, dpi=130)
    print("saved", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--data_root", default=None)
    ap.add_argument("--n", type=int, default=15)   # 3 per class across 5 classes
    a = ap.parse_args()
    main(a.config, a.checkpoint, a.data_root, a.n)
