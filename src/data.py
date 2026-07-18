"""
data.py — Scale x Odyssey leakage-safe data pipeline (5 classes, multi-source)
==============================================================================

Rebuilt to cover the full current dataset:

  class          sources (each a folder under DATA_ROOT)              origin rule
  -------------  ---------------------------------------------------  -----------
  nebula         spacenet_raw/nebula   + nasa_lib/nebula              spacenet / stem
  planet         spacenet_raw/planet   + nasa_lib/planet              spacenet / stem
  spiral         galaxy_zoo/spiral     + nasa_lib/spiral              stem / stem
  elliptical     galaxy_zoo/elliptical + nasa_lib/elliptical          stem / stem
  star_cluster   star_cluster          + nasa_lib/star_cluster        stem / stem

The two invariants that keep evaluation honest:
  1. ORIGIN, not file, is the unit of splitting. SpaceNet augmented variants of
     one original share an origin id and must never straddle train/val/test.
     A hard assertion crashes loudly if any origin appears in two splits.
  2. Split is stratified PER CLASS (we split each class's origins separately),
     so class balance is preserved across train/val/test.

Every record also carries its `source` folder, so you can later carve a held-out
OOD test set by source (e.g. the NASA-library imagery) — see split_on_origins'
`hold_out_source` argument. Default is None: all sources flow into the normal
split, which is what realises the cross-source diversification we built.

Public API:
  build_file_index(data_root) -> list[record]
  split_on_origins(records, ratios, seed, hold_out_source) -> {split: [record]}
  save_split / load_split
  AstroDataset(records, transform)          # PyTorch Dataset
  prepare_splits(...)                        # convenience: index -> split -> json
record = {"path", "class", "label", "origin", "source"}
"""

import os
import re
import json
import random
from collections import defaultdict, Counter

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
DATA_ROOT = "/content/drive/MyDrive/scale-odyssey/data"

LABELS = {
    "nebula": 0,
    "planet": 1,
    "spiral": 2,
    "elliptical": 3,
    "star_cluster": 4,
}

# class -> list of (relative_folder, origin_scheme). scheme in {"spacenet","stem"}.
SOURCES = {
    "nebula":       [("spacenet_raw/nebula",   "spacenet"), ("nasa_lib/nebula",       "stem")],
    "planet":       [("spacenet_raw/planet",   "spacenet"), ("nasa_lib/planet",       "stem")],
    "spiral":       [("galaxy_zoo/spiral",     "stem"),     ("nasa_lib/spiral",       "stem")],
    "elliptical":   [("galaxy_zoo/elliptical", "stem"),     ("nasa_lib/elliptical",   "stem")],
    "star_cluster": [("star_cluster",          "stem"),     ("nasa_lib/star_cluster", "stem")],
}

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

# SpaceNet naming:  ..._image_<N>[_<variant>]_SwinIR_large.png
#   <N>       = the image number (KEEP — this defines the original)
#   <variant> = an OPTIONAL augmentation suffix that comes AFTER the number,
#               either a single digit 0-4 or aug<digits>  (STRIP this)
# We anchor on 'image_<N>' and strip only a *following* variant, so:
#   - image_10 vs image_100 stay distinct (bounded variant digit), AND
#   - a single-digit image number (image_4) is NOT mistaken for a variant,
#     which previously split an original's base file from its own variants
#     (a real leakage bug — see the '..._image' thin origins in the audit).
SPACENET_RX = re.compile(
    r"^(.*_image_\d+)(?:_(?:[0-4]|aug\d+))?_SwinIR_large\.png$")


# --------------------------------------------------------------------------- #
# Origin extraction
# --------------------------------------------------------------------------- #
def get_origin_id(filename, scheme):
    """Return the *within-class* origin id for a filename.

    scheme="spacenet": strip the augmentation-variant suffix so all variants of
        one original collapse to a single origin.
    scheme="stem": the bare filename (no extension) — one image, one origin
        (Galaxy Zoo, star_cluster cutouts, NASA library).
    """
    if scheme == "spacenet":
        m = SPACENET_RX.match(filename)
        return m.group(1) if m else os.path.splitext(filename)[0]
    return os.path.splitext(filename)[0]


# --------------------------------------------------------------------------- #
# File index
# --------------------------------------------------------------------------- #
def build_file_index(data_root=DATA_ROOT, sources=SOURCES, labels=LABELS, verbose=True):
    """Walk every source folder -> list of records. Origins are namespaced by
    class ('<class>::<origin_id>') so ids never collide across classes."""
    records = []
    missing = []
    for cls, srcs in sources.items():
        label = labels[cls]
        for rel, scheme in srcs:
            folder = os.path.join(data_root, rel)
            if not os.path.isdir(folder):
                missing.append(rel)
                continue
            for fn in os.listdir(folder):
                if os.path.splitext(fn)[1].lower() not in IMG_EXTS:
                    continue
                oid = get_origin_id(fn, scheme)
                records.append({
                    "path":   os.path.join(folder, fn),
                    "class":  cls,
                    "label":  label,
                    "origin": f"{cls}::{oid}",   # class-namespaced
                    "source": rel,
                })
    if verbose:
        _print_index_summary(records, missing)
    return records


def _print_index_summary(records, missing):
    by_class = defaultdict(lambda: {"files": 0, "origins": set(), "sources": Counter()})
    for r in records:
        c = by_class[r["class"]]
        c["files"] += 1
        c["origins"].add(r["origin"])
        c["sources"][r["source"]] += 1
    print("=" * 68)
    print("FILE INDEX")
    print("=" * 68)
    print(f"{'class':<14}{'files':>7}{'origins':>9}   sources")
    for cls in SOURCES:
        if cls not in by_class:
            print(f"{cls:<14}{'--':>7}{'--':>9}   (none found)")
            continue
        c = by_class[cls]
        srcs = ", ".join(f"{k}:{v}" for k, v in c["sources"].items())
        print(f"{cls:<14}{c['files']:>7}{len(c['origins']):>9}   {srcs}")
    print(f"\nTOTAL files={len(records)}  "
          f"origins={len(set(r['origin'] for r in records))}")
    if missing:
        print(f"!! missing folders (skipped): {missing}")
    print("=" * 68)


# --------------------------------------------------------------------------- #
# Leakage-safe split (on origins, stratified per class)
# --------------------------------------------------------------------------- #
def split_on_origins(records, ratios=(0.70, 0.15, 0.15), seed=42,
                     hold_out_source=None):
    """Split records into train/val/test on ORIGINS, stratified per class.

    hold_out_source: if given (substring match on record['source']), all matching
        records are pulled into a separate 'ood' split and the rest are split
        normally. Use to reserve, say, the NASA-library imagery as an
        out-of-distribution test. Default None -> every source is split normally.

    Guarantees (asserted): no origin appears in more than one split.
    """
    assert abs(sum(ratios) - 1.0) < 1e-6, "ratios must sum to 1"

    ood = []
    main = []
    for r in records:
        if hold_out_source and hold_out_source in r["source"]:
            ood.append(r)
        else:
            main.append(r)

    origins_by_class = defaultdict(set)
    for r in main:
        origins_by_class[r["class"]].add(r["origin"])

    rng = random.Random(seed)
    origin_to_split = {}
    for cls in sorted(origins_by_class):
        origins = sorted(origins_by_class[cls])   # sort first -> deterministic
        rng.shuffle(origins)
        n = len(origins)
        n_tr = int(n * ratios[0])
        n_va = int(n * ratios[1])
        for i, o in enumerate(origins):
            origin_to_split[o] = ("train" if i < n_tr
                                  else "val" if i < n_tr + n_va
                                  else "test")

    splits = {"train": [], "val": [], "test": []}
    for r in main:
        splits[origin_to_split[r["origin"]]].append(r)
    if ood:
        splits["ood"] = ood

    _assert_no_leakage(splits)
    return splits


def _assert_no_leakage(splits):
    """Hard guarantee: an origin must live in exactly one split (ood excluded —
    it is a deliberately separate, held-out source)."""
    seen = {}
    for name, recs in splits.items():
        if name == "ood":
            continue
        for r in recs:
            prev = seen.get(r["origin"])
            assert prev in (None, name), (
                f"LEAKAGE: origin {r['origin']} in both '{prev}' and '{name}'")
            seen[r["origin"]] = name


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #
def save_split(splits, path="split.json"):
    meta = {
        "labels": LABELS,
        "counts": {k: len(v) for k, v in splits.items()},
        "origin_counts": {k: len(set(r["origin"] for r in v)) for k, v in splits.items()},
    }
    with open(path, "w") as f:
        json.dump({"meta": meta, "splits": splits}, f, indent=2)
    print(f"split saved -> {path}   {meta['counts']}")
    return path


def load_split(path="split.json"):
    with open(path) as f:
        return json.load(f)["splits"]


# --------------------------------------------------------------------------- #
# PyTorch Dataset (torch imported lazily so this module indexes without torch)
# --------------------------------------------------------------------------- #
def _dataset_base():
    try:
        from torch.utils.data import Dataset
        return Dataset
    except Exception:
        return object


class AstroDataset(_dataset_base()):
    """One class serves train/val/test. Fixed label mapping (LABELS).
    transform: a callable (e.g. torchvision transforms) applied to a PIL RGB image."""

    def __init__(self, records, transform=None):
        self.records = records
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        from PIL import Image
        r = self.records[idx]
        img = Image.open(r["path"]).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, r["label"]


# --------------------------------------------------------------------------- #
# Convenience end-to-end
# --------------------------------------------------------------------------- #
def prepare_splits(data_root=DATA_ROOT, out_json="split.json",
                   ratios=(0.70, 0.15, 0.15), seed=42, hold_out_source=None):
    index = build_file_index(data_root)
    splits = split_on_origins(index, ratios=ratios, seed=seed,
                              hold_out_source=hold_out_source)
    print("\nsplit sizes (files):",
          {k: len(v) for k, v in splits.items()})
    print("split sizes (origins):",
          {k: len(set(r["origin"] for r in v)) for k, v in splits.items()})
    # per-class per-split origin counts (confirms stratification)
    print("\nper-class origins by split:")
    for cls in SOURCES:
        row = {s: len(set(r["origin"] for r in recs if r["class"] == cls))
               for s, recs in splits.items()}
        print(f"  {cls:<14} {row}")
    save_split(splits, out_json)
    return splits


if __name__ == "__main__":
    # Default: fold every source into a normal stratified split (realises the
    # cross-source diversification). To reserve NASA imagery as an OOD test:
    #   prepare_splits(hold_out_source="nasa_lib")
    prepare_splits()
