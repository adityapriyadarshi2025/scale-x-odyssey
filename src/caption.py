"""
caption.py — image captioning (Phase 4 bonus).

Generates a short natural-language description of an astronomical image, e.g.
"A spiral galaxy with a bright central core and extended spiral arms."

Approach. The caption is composed from the classifier's own prediction plus a few
interpretable image measurements — dominant colour, whether there is a
concentrated bright core, the object's shape/elongation, how much of the frame it
fills, and a star-richness proxy for clusters. This is deliberate: a generic
pretrained caption model (trained on everyday photos) hallucinates on astronomical
imagery, whereas a class-conditioned template driven by *measured* attributes
stays accurate and defensible.

Run from repo root:
    python -m src.caption --checkpoint <best_model.pth> --data_root <data> --n 10
"""

import os
import argparse

import numpy as np
import torch
from PIL import Image

from . import transforms as T
from . import eval as E


def _attributes(rgb):
    """Measure interpretable attributes from an HxWx3 float image in [0,1]."""
    g = rgb @ np.array([0.299, 0.587, 0.114])
    mr, mg, mb = rgb[..., 0].mean(), rgb[..., 1].mean(), rgb[..., 2].mean()
    if mr > mb * 1.12 and mr > mg * 1.05:
        colour = "reddish"
    elif mb > mr * 1.12 and mg > mr * 1.08 and abs(mb - mg) < 0.35 * max(mb, mg):
        colour = "teal"                       # blue-green (OIII / planetary nebulae)
    elif mb > mr * 1.12 and mb > mg * 1.05:
        colour = "blue-hued"
    elif mg > mr * 1.1 and mg > mb * 1.1:
        colour = "green-hued"
    else:
        colour = "multicoloured"

    peak = float(g.max()) + 1e-6
    bright = g > 0.6 * peak
    frac = float(bright.mean())
    core, shape = False, "round"
    ys, xs = np.where(bright)
    if len(xs) > 5:
        H, W = g.shape
        cx, cy = xs.mean(), ys.mean()
        cov = np.cov(np.vstack([xs, ys]))
        ev = np.sort(np.linalg.eigvalsh(cov))[::-1]
        shape = "elongated" if (ev[0] / (ev[1] + 1e-6)) ** 0.5 > 1.7 else "round"
        core = frac < 0.12 and abs(cx - W / 2) < 0.28 * W and abs(cy - H / 2) < 0.28 * H

    try:
        from scipy import ndimage
        _, nstar = ndimage.label(g > 0.75 * peak)
    except Exception:
        nstar = 30 if (g > 0.75 * peak).sum() > 60 else 5
    return colour, core, shape, frac, nstar


def caption_for(cls, attrs, conf=None):
    """Compose a caption from MEASURED attributes. Every adjective traces to a
    measurement (dominant colour, extent, central concentration, elongation, star
    richness) — nothing is invented. If conf is low, the caption hedges."""
    colour, core, shape, frac, nstar = attrs
    ext = " filling much of the frame" if frac > 0.30 else " set against dark space"

    if cls == "spiral":
        if core and colour == "blue-hued":
            t = "A blue, actively star-forming spiral galaxy with a bright core and sweeping arms."
        elif core:
            t = "A spiral galaxy with a bright central core and winding arms."
        elif colour == "reddish":
            t = "An older, reddish spiral galaxy with a soft, diffuse centre."
        else:
            t = "A spiral galaxy with arms winding outward from its centre."
    elif cls == "elliptical":
        col = "" if colour == "multicoloured" else colour + " "
        t = (f"An elongated, cigar-shaped elliptical galaxy — a smooth {col}glow."
             if shape == "elongated"
             else f"A nearly round elliptical galaxy — a smooth, featureless {col}glow.")
    elif cls == "nebula":
        if colour == "reddish":
            t = "A reddish emission nebula, glowing with ionised hydrogen" + ext + "."
        elif colour == "blue-hued":
            t = "A blue reflection nebula — starlight scattered off surrounding dust" + ext + "."
        elif colour in ("teal", "green-hued") and frac < 0.15:
            t = "A compact planetary nebula — a glowing shell of gas cast off by a dying star."
        elif colour in ("teal", "green-hued"):
            t = "A teal nebula of glowing ionised gas" + ext + "."
        else:
            t = f"A {colour} nebula, a diffuse cloud of gas and dust" + ext + "."
    elif cls == "star_cluster":
        dense = nstar > 25 or core
        t = ("A dense, spherical globular cluster — many stars concentrated toward a bright core."
             if dense else "A loose, scattered open cluster of stars.")
    elif cls == "planet":
        t = ("A planetary object showing a bright, sharply defined disc." if core
             else "A planetary object showing a round disc against dark space.")
    else:
        t = f"A {cls}."

    if conf is not None and conf < 0.60:
        t = "Possibly " + t[0].lower() + t[1:]
    return t


def caption_image(model, pil, cfg, device):
    d = cfg["data"]; size = d["image_size"]
    tf = T.eval_transforms(size, d["norm_mean"], d["norm_std"])
    x = tf(pil.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(x), 1)[0]
    pred = int(probs.argmax()); conf = float(probs[pred])
    rgb = np.asarray(pil.convert("RGB").resize((size, size)), dtype=np.float32) / 255.0
    cap = caption_for(cfg["classes"][pred], _attributes(rgb), conf)
    return cap, cfg["classes"][pred], conf, rgb


def main(config, ckpt, data_root, n):
    import matplotlib.pyplot as plt
    from collections import defaultdict
    cfg = E.load_config(config)
    if data_root:
        cfg["paths"]["data_root"] = data_root
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = E.load_model(cfg, ckpt or os.path.join(cfg["paths"]["artifacts_dir"], "best_model.pth"), device)

    by = defaultdict(list)
    for r in E.build_splits(cfg)["test"]:
        by[r["class"]].append(r)
    per = max(1, n // len(cfg["classes"]))
    recs = [r for c in cfg["classes"] for r in by[c][:per]][:n]

    cols = 2
    rows = (len(recs) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 3 * rows))
    axes = np.array(axes).reshape(-1)
    for i, r in enumerate(recs):
        cap, pred, conf, rgb = caption_image(model, Image.open(r["path"]), cfg, device)
        ax = axes[i]; ax.imshow(rgb); ax.axis("off")
        ax.set_title(f'"{cap}"\n[{pred} · {conf:.0%}]', fontsize=8)
        print(f"[{pred:<12}] {cap}")
    for j in range(len(recs), len(axes)):
        axes[j].axis("off")
    fig.tight_layout()
    out = os.path.join(cfg["paths"]["artifacts_dir"], "captions.png")
    os.makedirs(cfg["paths"]["artifacts_dir"], exist_ok=True)
    fig.savefig(out, dpi=130)
    print("saved", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--data_root", default=None)
    ap.add_argument("--n", type=int, default=10)
    a = ap.parse_args()
    main(a.config, a.checkpoint, a.data_root, a.n)
