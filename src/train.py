"""
train.py — fine-tune EfficientNet-B0 on the 5-class split.
Run:  python -m src.train --config configs/default.yaml
"""

import os
import random
import argparse
from collections import Counter

import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from . import data as D
from . import transforms as T
from . import model as M


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def seed_everything(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def class_weights(records, n_classes):
    c = Counter(r["label"] for r in records)
    counts = [c.get(i, 0) for i in range(n_classes)]
    total = sum(counts)
    return torch.tensor(
        [total / (n_classes * n) if n else 0.0 for n in counts], dtype=torch.float)


def make_loaders(cfg, splits):
    d = cfg["data"]
    train_tf, eval_tf = T.get_transforms(
        image_size=d["image_size"], mean=d["norm_mean"], std=d["norm_std"],
        strength=cfg["train"].get("aug_strength", "standard"))
    bs, nw = cfg["train"]["batch_size"], cfg["train"]["num_workers"]
    train_ld = DataLoader(D.AstroDataset(splits["train"], train_tf),
                          batch_size=bs, shuffle=True, num_workers=nw, pin_memory=True)
    val_ld = DataLoader(D.AstroDataset(splits["val"], eval_tf),
                        batch_size=bs, shuffle=False, num_workers=nw, pin_memory=True)
    return train_ld, val_ld


def run_epoch(model, loader, criterion, device, optimizer=None):
    train = optimizer is not None
    model.train() if train else model.eval()
    tot_loss, correct, n = 0.0, 0, 0
    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            if train:
                optimizer.zero_grad(); loss.backward(); optimizer.step()
            tot_loss += loss.item() * x.size(0)
            correct += (out.argmax(1) == y).sum().item()
            n += x.size(0)
    return tot_loss / n, correct / n


def main(config_path, data_root=None):
    cfg = load_config(config_path)
    if data_root:                       # override Drive path with a fast local copy
        cfg["paths"]["data_root"] = data_root
    seed_everything(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    records = D.build_file_index(cfg["paths"]["data_root"])
    splits = D.split_on_origins(records, ratios=cfg["split"]["ratios"],
                                seed=cfg["seed"], hold_out_source=cfg["split"]["hold_out_source"])
    os.makedirs(cfg["paths"]["artifacts_dir"], exist_ok=True)
    D.save_split(splits, cfg["paths"]["split_json"])

    train_ld, val_ld = make_loaders(cfg, splits)
    model = M.from_config(cfg).to(device)

    n_classes = len(cfg["classes"])
    w = class_weights(splits["train"], n_classes).to(device) if cfg["train"]["class_weighting"] else None
    criterion = nn.CrossEntropyLoss(weight=w, label_smoothing=cfg["train"]["label_smoothing"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"],
                                  weight_decay=cfg["train"]["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["train"]["epochs"])

    use_wandb = cfg["wandb"]["enabled"]
    if use_wandb:
        try:
            import wandb
            wandb.init(project=cfg["wandb"]["project"], config=cfg)
        except Exception as e:
            print("wandb disabled:", e); use_wandb = False

    best_acc, patience, since_best = 0.0, cfg["train"]["early_stop_patience"], 0
    ckpt = os.path.join(cfg["paths"]["artifacts_dir"], "best_model.pth")

    for epoch in range(1, cfg["train"]["epochs"] + 1):
        tr_loss, tr_acc = run_epoch(model, train_ld, criterion, device, optimizer)
        va_loss, va_acc = run_epoch(model, val_ld, criterion, device)
        scheduler.step()
        lr = optimizer.param_groups[0]["lr"]
        print(f"epoch {epoch:02d}  train {tr_acc:.3f}/{tr_loss:.3f}  "
              f"val {va_acc:.3f}/{va_loss:.3f}  lr {lr:.2e}")
        if use_wandb:
            import wandb
            wandb.log({"epoch": epoch, "train_acc": tr_acc, "train_loss": tr_loss,
                       "val_acc": va_acc, "val_loss": va_loss, "lr": lr})

        if va_acc > best_acc:
            best_acc, since_best = va_acc, 0
            torch.save({"model": model.state_dict(), "epoch": epoch,
                        "val_acc": va_acc, "classes": cfg["classes"]}, ckpt)
            print(f"  ✓ new best {best_acc:.3f} -> {ckpt}")
        else:
            since_best += 1
            if since_best >= patience:
                print(f"early stop (no val gain in {patience} epochs)"); break

    print(f"done. best val acc = {best_acc:.3f}")
    return best_acc


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--data_root", default=None,
                    help="override data path (e.g. a fast local /content copy)")
    a = ap.parse_args()
    main(a.config, a.data_root)
