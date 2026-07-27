"""
fetch_galaxies_legacy.py — boost spiral/elliptical with real DECaLS galaxy images.

Why. spiral/elliptical are the model's weakest classes (elliptical F1 0.81) and
the most data-starved (~200 origins vs 1,200+ for nebula/planet). This adds a
THIRD source for them from a genuinely different instrument — DESI Legacy Survey
(DECaLS) — so it's both a quantity boost AND cross-instrument diversity, not more
of the same SDSS-cutout Galaxy Zoo imagery.

Source. Galaxy10 DECaLS (17,736 DECaLS galaxy images, Galaxy Zoo morphology
labels; astroNN / Bowles et al.). One .h5 file with 'images' (Nx256x256x3 uint8)
and 'ans' (0-9 class labels).

Label map (Galaxy10 10-class -> our 2 galaxy classes):
  elliptical <- 2 Round Smooth, 3 In-between Smooth, 4 Cigar-shaped Smooth
  spiral     <- 5 Barred Spiral, 6 Tight Spiral, 7 Loose Spiral
  excluded   <- 0 Disturbed, 1 Merging, 8/9 Edge-on  (ambiguous for our 2 classes)

Output. data/legacy_gal/<spiral|elliptical>/gal_<idx>.jpg  (one image = one origin).
Add these two folders as a THIRD source for spiral/elliptical in data.py.

Requires: h5py, requests, pillow, numpy. If the direct download URL ever breaks,
fall back to:  !pip install astroNN  then  from astroNN.datasets import load_galaxy10
"""

import os
import numpy as np
from PIL import Image

OUT_ROOT = "/content/so_data/legacy_gal"
H5_PATH  = "/content/Galaxy10_DECals.h5"
# astroNN-hosted mirror (update if it 404s — see astroNN fallback in the docstring)
H5_URL   = "https://www.astro.utoronto.ca/~hleung/shared/Galaxy10/Galaxy10_DECals.h5"

TARGET_PER_CLASS = 700          # brings each galaxy class to ~950 files (200 GZ + 50 NASA + 700)
ELLIPTICAL = {2, 3, 4}
SPIRAL     = {5, 6, 7}


def _download():
    if os.path.exists(H5_PATH) and os.path.getsize(H5_PATH) > 1_000_000_000:
        print("Galaxy10 h5 already present.")
        return
    import requests
    print("downloading Galaxy10 DECaLS (~2.5 GB, a few minutes)...")
    with requests.get(H5_URL, stream=True, timeout=180) as r:
        r.raise_for_status()
        with open(H5_PATH, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
    print(f"downloaded {os.path.getsize(H5_PATH)/1e6:.0f} MB")


def build(target=TARGET_PER_CLASS, seed=42):
    import h5py
    _download()
    with h5py.File(H5_PATH, "r") as f:
        labels = np.array(f["ans"])
        images = f["images"]                        # lazy; index selectively
        rng = np.random.default_rng(seed)
        summary = {}
        for cls, ids in [("elliptical", ELLIPTICAL), ("spiral", SPIRAL)]:
            idx = np.where(np.isin(labels, list(ids)))[0]
            rng.shuffle(idx)
            idx = np.sort(idx[:target])              # h5py fancy-index needs sorted
            arrs = images[idx]                       # bulk load only what we keep
            out = os.path.join(OUT_ROOT, cls)
            os.makedirs(out, exist_ok=True)
            for i, arr in zip(idx, arrs):
                Image.fromarray(arr).convert("RGB").save(
                    os.path.join(out, f"gal_{int(i)}.jpg"), quality=92)
            summary[cls] = len(idx)
            print(f"{cls:<11} saved {len(idx)}  -> {out}")

    print("\n=== DECaLS galaxy boost ===")
    for c, n in summary.items():
        print(f"  {c:<11} +{n}")
    print(f"root -> {OUT_ROOT}")
    print("NEXT: add legacy_gal/spiral & legacy_gal/elliptical as a 3rd source "
          "in data.py, then retrain.")
    return summary


if __name__ == "__main__":
    build()
