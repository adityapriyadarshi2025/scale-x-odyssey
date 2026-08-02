"""
fetch_galaxies_spacetel.py — boost SPACE-TELESCOPE galaxies for training
========================================================================

Why. Diagnostics on real Hubble imagery showed the galaxy classes fail on
space-telescope images: the model is colour-brittle AND its ~50 NASA/Hubble
galaxies are outnumbered ~18:1 by ground-based survey cutouts (Galaxy Zoo +
DECaLS). This pulls MORE Hubble *and JWST* spirals/ellipticals so the
cross-instrument examples are no longer drowned out.

Source. NASA Image and Video Library (images-api.nasa.gov) — public, no key,
real Hubble/JWST processed RGB. Wider queries than the original pass: adds
JWST/Webb and specific famous galaxies to surface more DISTINCT objects.

OOD INTEGRITY (the important part). Your out-of-distribution test set is ESA
Hubble galaxies. To keep the before/after honest, every candidate is checked
against data/ood/ with a perceptual average-hash (aHash) and DROPPED if it
looks like any OOD image — so no test galaxy can leak into training, even if
the same famous object (M51, M87...) appears in both crawls under different files.

Output. data/spacetel_gal/<spiral|elliptical>/<id>.jpg  (+ manifest.csv).
Kept in its OWN folder for clean provenance; wire into data.py as a THIRD
galaxy source (one-line edit, see bottom). Idempotent: re-runs skip existing.

Run in Colab (Drive mounted):
    python fetch_galaxies_spacetel.py            # smoke test grid first
    # then in a cell:  from fetch_galaxies_spacetel import build; build()

Requires: requests, pillow, matplotlib (all present in Colab).
"""

import os
import io
import csv
import time
import re
import hashlib

import requests
from PIL import Image

# --------------------------------------------------------------------------- #
DATA_ROOT = "/content/drive/MyDrive/scale-odyssey/data"
OUT_ROOT  = os.path.join(DATA_ROOT, "spacetel_gal")
OOD_ROOT  = os.path.join(DATA_ROOT, "ood")          # to protect the test set

API_SEARCH = "https://images-api.nasa.gov/search"
API_ASSET  = "https://images-api.nasa.gov/asset/{}"

TARGET_PER_CLASS = 250
MAX_PAGES        = 15
MAX_ASPECT       = 2.2
MIN_SIDE_PX      = 200
AHASH_HAMMING    = 6          # <= this distance to an OOD image => treat as leak, drop
PREFERRED_SIZES  = ["~medium.jpg", "~large.jpg", "~orig.jpg", "~small.jpg"]

REQUEST_PAUSE = 0.5
RETRIES       = 3
TIMEOUT       = 60

# Wider, instrument-diverse + named-object queries to surface DISTINCT galaxies.
CLASS_QUERIES = {
    "spiral": [
        "spiral galaxy Hubble", "barred spiral galaxy Hubble",
        "spiral galaxy JWST", "spiral galaxy Webb", "face-on spiral galaxy",
        "grand design spiral galaxy", "Messier spiral galaxy",
        "NGC spiral galaxy", "Whirlpool galaxy", "Pinwheel galaxy",
        "Sombrero galaxy", "Andromeda galaxy Hubble", "Triangulum galaxy",
    ],
    "elliptical": [
        "elliptical galaxy Hubble", "elliptical galaxy JWST",
        "lenticular galaxy Hubble", "giant elliptical galaxy",
        "Messier 87 galaxy", "Centaurus A galaxy", "M60 galaxy",
        "M49 galaxy", "NGC elliptical galaxy", "cD galaxy Hubble",
    ],
}

# Drop non-photographic results and, importantly, galaxy-CLUSTER shots
# (a field of many galaxies) which aren't a single spiral/elliptical.
JUNK_RX = re.compile(
    r"\b(illustration|artist|concept|rendering|render|animation|diagram|"
    r"infographic|chart|graph|map|poster|logo|schematic|simulation|artwork|"
    r"impression|cutaway|timeline|comparison|labell?ed|deep field|"
    r"galaxy cluster|cluster of galaxies|hubble ultra deep|survey field)\b",
    re.IGNORECASE,
)


# ------------------------- perceptual hash (aHash) ------------------------- #
def ahash(img):
    g = img.convert("L").resize((8, 8))
    px = list(g.getdata())
    avg = sum(px) / len(px)
    bits = 0
    for i, p in enumerate(px):
        if p >= avg:
            bits |= (1 << i)
    return bits


def hamming(a, b):
    return bin(a ^ b).count("1")


def load_ood_hashes():
    """aHash every OOD image so we can reject look-alikes from training."""
    hashes = []
    if not os.path.isdir(OOD_ROOT):
        print("  (no ood/ folder found — skipping leak guard; verify manually!)")
        return hashes
    for root, _, files in os.walk(OOD_ROOT):
        for f in files:
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                try:
                    hashes.append(ahash(Image.open(os.path.join(root, f)).convert("RGB")))
                except Exception:
                    pass
    print(f"  loaded {len(hashes)} OOD aHashes for leak protection")
    return hashes


def is_ood_lookalike(img, ood_hashes):
    h = ahash(img)
    return any(hamming(h, o) <= AHASH_HAMMING for o in ood_hashes)


# ----------------------------- API helpers -------------------------------- #
def _get_json(url, params=None):
    for attempt in range(1, RETRIES + 1):
        try:
            r = requests.get(url, params=params, timeout=TIMEOUT)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            if attempt == RETRIES:
                print(f"    api error {url}: {e}")
                return None
            time.sleep(REQUEST_PAUSE * attempt)


def search_ids(query, want, seen_titles):
    out, page = [], 1
    while len(out) < want and page <= MAX_PAGES:
        js = _get_json(API_SEARCH, {"q": query, "media_type": "image", "page": page})
        if not js:
            break
        items = js.get("collection", {}).get("items", [])
        if not items:
            break
        for it in items:
            d = (it.get("data") or [{}])[0]
            nid = d.get("nasa_id")
            title = (d.get("title") or "").strip()
            blob = f"{title} {' '.join(d.get('keywords', []))} {d.get('description','')}"
            if not nid or not title or JUNK_RX.search(blob):
                continue
            key = title.lower()
            if key in seen_titles:
                continue
            seen_titles.add(key)
            out.append((nid, title))
            if len(out) >= want:
                break
        page += 1
        time.sleep(REQUEST_PAUSE)
    return out


def asset_image_url(nasa_id):
    js = _get_json(API_ASSET.format(requests.utils.quote(nasa_id)))
    if not js:
        return None
    hrefs = [i.get("href", "") for i in js.get("collection", {}).get("items", [])]
    hrefs = [h for h in hrefs if h.lower().endswith((".jpg", ".jpeg", ".png"))]
    for size in PREFERRED_SIZES:
        for h in hrefs:
            if h.endswith(size):
                return h
    return hrefs[0] if hrefs else None


def fetch_image(url):
    for attempt in range(1, RETRIES + 1):
        try:
            r = requests.get(url, timeout=TIMEOUT)
            r.raise_for_status()
            im = Image.open(io.BytesIO(r.content)).convert("RGB")
            w, h = im.size
            if min(w, h) < MIN_SIDE_PX:
                raise ValueError(f"too small {im.size}")
            if max(w, h) / min(w, h) > MAX_ASPECT:
                raise ValueError(f"extreme aspect {im.size}")
            if max(hi for _, hi in im.getextrema()) < 8:
                raise ValueError("blank/near-black")
            return im
        except Exception as e:
            if attempt == RETRIES:
                print(f"    fetch failed: {e}  <{url[:70]}>")
                return None
            time.sleep(REQUEST_PAUSE * attempt)


# ------------------------------ smoke test -------------------------------- #
def smoke_test(per_class=5):
    import matplotlib.pyplot as plt
    ood_hashes = load_ood_hashes()
    classes = list(CLASS_QUERIES.keys())
    fig, axes = plt.subplots(len(classes), per_class,
                             figsize=(3 * per_class, 3 * len(classes)))
    for i, cls in enumerate(classes):
        seen, ids = set(), []
        for q in CLASS_QUERIES[cls]:
            ids += search_ids(q, per_class, seen)
            if len(ids) >= per_class:
                break
        for j in range(per_class):
            ax = axes[i][j] if per_class > 1 else axes[i]
            ax.axis("off")
            if j < len(ids):
                url = asset_image_url(ids[j][0])
                im = fetch_image(url) if url else None
                if im is not None:
                    tag = "LEAK!" if is_ood_lookalike(im, ood_hashes) else "ok"
                    ax.imshow(im)
                    ax.set_title(f"{cls} [{tag}]\n{ids[j][1][:22]}", fontsize=7)
            time.sleep(REQUEST_PAUSE)
    plt.tight_layout(); plt.show()
    print("\nCheck: real single galaxies of the right type? Any 'LEAK!' is an OOD "
          "look-alike that build() will auto-drop. Tune CLASS_QUERIES if needed.")


# ------------------------------- full pull -------------------------------- #
def build(target_per_class=TARGET_PER_CLASS, only=None):
    classes = only or list(CLASS_QUERIES.keys())
    ood_hashes = load_ood_hashes()
    grand, leaks_blocked = {}, 0
    for cls in classes:
        out_dir = os.path.join(OUT_ROOT, cls)
        os.makedirs(out_dir, exist_ok=True)
        manifest = os.path.join(out_dir, "manifest.csv")

        existing = set()
        for f in os.listdir(out_dir):
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                try:
                    existing.add(hashlib.md5(open(os.path.join(out_dir, f), "rb").read()).hexdigest())
                except Exception:
                    pass
        have0 = len(existing)
        print(f"\n### {cls}  (target {target_per_class}, already have {have0})")

        seen_titles, saved, rows = set(), 0, []
        for q in CLASS_QUERIES[cls]:
            if have0 + saved >= target_per_class:
                break
            need = target_per_class - have0 - saved
            for nid, title in search_ids(q, need * 2, seen_titles):
                if have0 + saved >= target_per_class:
                    break
                url = asset_image_url(nid)
                if not url:
                    continue
                im = fetch_image(url)
                if im is None:
                    continue
                if is_ood_lookalike(im, ood_hashes):      # <-- protect the OOD test
                    leaks_blocked += 1
                    continue
                digest = hashlib.md5(im.tobytes()).hexdigest()
                if digest in existing:
                    continue
                existing.add(digest)
                safe = re.sub(r"[^0-9A-Za-z_-]", "", nid)[:40] or f"img{saved}"
                fname = f"{cls}_st_{safe}.jpg"
                im.save(os.path.join(out_dir, fname), quality=92)
                rows.append([fname, cls, nid, title, q])
                saved += 1
                if saved % 20 == 0:
                    print(f"    ...{saved} saved")
                time.sleep(REQUEST_PAUSE)

        write_header = not os.path.exists(manifest)
        with open(manifest, "a", newline="") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(["file", "class", "nasa_id", "title", "query"])
            w.writerows(rows)
        grand[cls] = saved
        print(f"  {cls}: +{saved} new -> {out_dir}")

    print("\n=== space-telescope galaxy boost summary ===")
    for cls, n in grand.items():
        print(f"  {cls:<12} +{n}")
    print(f"  OOD look-alikes blocked (kept out of training): {leaks_blocked}")
    print(f"Root -> {OUT_ROOT}")
    print("\nNEXT: add spacetel_gal/<cls> as a THIRD galaxy source in data.py "
          "SOURCES for spiral & elliptical, then re-run audit_data.py and retrain.")


if __name__ == "__main__":
    smoke_test()
