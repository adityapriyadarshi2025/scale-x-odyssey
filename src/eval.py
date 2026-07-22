"""
eval.py — evaluate a trained checkpoint: metrics, confusion matrix, Grad-CAM, OOD.
Run:  python -m src.eval --checkpoint <path> --data_root /content/so_data
"""

import os
import argparse

import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader

from . import data as D
from . import transforms as T
from . import model as M


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def build_splits(cfg):
    records = D.build_file_index(cfg["paths"]["data_root"], verbose=False)
    return D.split_on_origins(records, ratios=cfg["split"]["ratios"],
                              seed=cfg["seed"], hold_out_source=cfg["split"]["hold_out_source"])


def load_model(cfg, ckpt_path, device):
    model = M.from_config(cfg)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model"])
    print(f"loaded {ckpt_path}  (epoch {state.get('epoch','?')}, "
          f"val_acc {state.get('val_acc','?')})")
    return model.to(device).eval()


@torch.no_grad()
def predict(model, loader, device):
    ys, ps = [], []
    for x, y in loader:
        ps.append(model(x.to(device)).argmax(1).cpu())
        ys.append(y)
    return torch.cat(ys).numpy(), torch.cat(ps).numpy()


def evaluate_split(model, records, cfg, device, name):
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
    d = cfg["data"]
    tf = T.eval_transforms(d["image_size"], d["norm_mean"], d["norm_std"])
    ld = DataLoader(D.AstroDataset(records, tf), batch_size=cfg["train"]["batch_size"],
                    shuffle=False, num_workers=cfg["train"]["num_workers"])
    y, p = predict(model, ld, device)
    classes = cfg["classes"]
    acc = accuracy_score(y, p)
    print(f"\n===== {name.upper()}  (n={len(y)})  accuracy = {acc:.4f} =====")
    print(classification_report(y, p, labels=list(range(len(classes))),
                                target_names=classes, digits=3, zero_division=0))
    cm = confusion_matrix(y, p, labels=list(range(len(classes))))
    _save_confusion(cm, classes, cfg, name)
    return acc, cm


def _save_confusion(cm, classes, cfg, name):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.5, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes))); ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right"); ax.set_yticklabels(classes)
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(f"Confusion — {name}")
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im); fig.tight_layout()
    out = os.path.join(cfg["paths"]["artifacts_dir"], f"confusion_{name}.png")
    fig.savefig(out, dpi=130); plt.close(fig)
    print("saved", out)


def _target_layer(model):
    for attr in ("conv_head",):
        if hasattr(model, attr):
            return getattr(model, attr)
    return model.blocks[-1]          # fallback for timm efficientnet


def gradcam(model, records, cfg, device, n=8):
    try:
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.image import show_cam_on_image
    except Exception as e:
        print("grad-cam unavailable:", e); return
    import matplotlib.pyplot as plt
    from PIL import Image
    d = cfg["data"]; size = d["image_size"]
    tf = T.eval_transforms(size, d["norm_mean"], d["norm_std"])
    cam = GradCAM(model=model, target_layers=[_target_layer(model)])

    sample = records[:n]
    fig, axes = plt.subplots(2, len(sample), figsize=(2.2 * len(sample), 4.6))
    for k, r in enumerate(sample):
        img = Image.open(r["path"]).convert("RGB")
        inp = tf(img).unsqueeze(0).to(device)
        rgb = np.array(img.resize((size, size))).astype(np.float32) / 255.0
        heat = cam(input_tensor=inp)[0]
        vis = show_cam_on_image(rgb, heat, use_rgb=True)
        axes[0][k].imshow(rgb); axes[0][k].set_title(r["class"], fontsize=7); axes[0][k].axis("off")
        axes[1][k].imshow(vis); axes[1][k].axis("off")
    fig.suptitle("Grad-CAM (top: input, bottom: attention)")
    fig.tight_layout()
    out = os.path.join(cfg["paths"]["artifacts_dir"], "gradcam.png")
    fig.savefig(out, dpi=130); plt.close(fig)
    print("saved", out)


def main(config_path, checkpoint=None, data_root=None):
    cfg = load_config(config_path)
    if data_root:
        cfg["paths"]["data_root"] = data_root
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = checkpoint or os.path.join(cfg["paths"]["artifacts_dir"], "best_model.pth")
    os.makedirs(cfg["paths"]["artifacts_dir"], exist_ok=True)

    model = load_model(cfg, ckpt, device)
    splits = build_splits(cfg)

    evaluate_split(model, splits["test"], cfg, device, "test")
    if "ood" in splits:
        evaluate_split(model, splits["ood"], cfg, device, "ood")
    gradcam(model, splits["test"], cfg, device)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--data_root", default=None)
    a = ap.parse_args()
    main(a.config, a.checkpoint, a.data_root)
