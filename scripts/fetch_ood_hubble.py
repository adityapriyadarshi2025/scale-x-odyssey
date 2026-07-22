"""
fetch_ood_hubble.py — build a held-out OOD test set from real ESA/Hubble imagery.

Purpose. The model trains on SpaceNet astrophotography, SDSS/survey cutouts, and
a small NASA-library slice. A separate set of real ESA Hubble press images — a
different telescope and processing pipeline it has never seen — measures true
out-of-distribution generalisation. This is the analysis-rubric's OOD test, and
it is NOT used for training.

Source. The 'Supermaxman/esa-hubble' Hugging Face dataset (a crawl of ESA Hubble
public images with an object 'Type' field). Loads via `datasets`, no scraping.

Labelling. ESA Hubble's own 'Type'/'Category'/title text is mapped to our five
classes. Careful cases:
  - "planetary nebula" -> nebula   (checked before 'planet')
  - "galaxy cluster"   -> skipped  (a cluster of galaxies, not a star cluster)
  - generic/irregular/interacting galaxies -> skipped (can't call spiral vs elliptical)
Ambiguous rows are dropped rather than guessed.

Output. data/ood/<class>/ood_<id>.jpg  + manifest.csv.  Evaluate with the cell
printed at the end (reuses eval.py). Requires:  pip install datasets
"""

import os
import csv
import re
import hashlib

from PIL import Image

OUT_ROOT   = "/content/so_data/ood"     # local; sync to Drive afterwards
CAP        = 60                          # max kept per class (dataset is small anyway)
MIN_SIDE   = 200                         # drop tiny images
MAX_ASPECT = 2.4                         # drop extreme panoramas
SAVE_MAXPX = 1024                        # downscale long side on save (eval resizes to 224)


def map_class(*texts):
    """Map ESA Hubble Type/Category/title text -> one of our 5 classes, or None."""
    t = " ".join(x for x in texts if x).lower()
    if "spiral" in t:
        return "spiral"
    if "elliptical" in t or "lenticular" in t:
        return "elliptical"
    if "nebula" in t:                       # planetary nebula lands here, correctly
        return "nebula"
    if "cluster" in t and "galax" not in t:  # star/globular/open cluster, not galaxy cluster
        return "star_cluster"
    if re.search(r"\b(planet|jupiter|saturn|mars|neptune|uranus|venus|solar system)\b", t):
        return "planet"
    return None


def _downscale(img):
    w, h = img.size
    m = max(w, h)
    if m > SAVE_MAXPX:
        s = SAVE_MAXPX / m
        img = img.resize((int(w * s), int(h * s)))
    return img


def build(cap=CAP):
    from datasets import load_dataset
    ds = load_dataset("Supermaxman/esa-hubble", split="train")
    print("loaded", len(ds), "ESA Hubble records; columns:", ds.column_names)

    seen_px = set()                          # within-set dedup by pixel hash
    counts, rows = {}, []
    for i, ex in enumerate(ds):
        cls = map_class(ex.get("Type", ""), ex.get("Category", ""), ex.get("title", ""))
        if cls is None or counts.get(cls, 0) >= cap:
            continue
        img = ex.get("image")
        if img is None:
            continue
        img = img.convert("RGB")
        w, h = img.size
        if min(w, h) < MIN_SIDE or max(w, h) / min(w, h) > MAX_ASPECT:
            continue
        digest = hashlib.md5(img.tobytes()).hexdigest()
        if digest in seen_px:
            continue
        seen_px.add(digest)

        img = _downscale(img)
        out_dir = os.path.join(OUT_ROOT, cls)
        os.makedirs(out_dir, exist_ok=True)
        iid = re.sub(r"[^0-9A-Za-z_-]", "", str(ex.get("id") or ex.get("Id") or i))
        img.save(os.path.join(out_dir, f"ood_{iid}.jpg"), quality=92)
        counts[cls] = counts.get(cls, 0) + 1
        rows.append([cls, iid, ex.get("Type", ""), (ex.get("title", "") or "")[:80]])

    os.makedirs(OUT_ROOT, exist_ok=True)
    with open(os.path.join(OUT_ROOT, "manifest.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class", "id", "type", "title"])
        w.writerows(rows)

    print("\n=== OOD (ESA Hubble) per-class counts ===")
    for c in ["nebula", "planet", "spiral", "elliptical", "star_cluster"]:
        print(f"  {c:<13} {counts.get(c, 0)}")
    print(f"total {sum(counts.values())}  ->  {OUT_ROOT}")
    return counts


if __name__ == "__main__":
    build()
