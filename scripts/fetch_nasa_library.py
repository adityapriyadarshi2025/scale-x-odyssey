"""
fetch_nasa_library.py — diversify ALL 5 classes with a second, real source
===========================================================================

Why: four of our classes are effectively single-source (nebula/planet =
SpaceNet, spiral/elliptical = Galaxy Zoo, star_cluster = survey cutouts). A CNN
can latch onto the *fingerprint of a source* (colour calibration, noise, PSF)
instead of the object — the "source-signature shortcut" — and then collapse on a
test set drawn from different instruments. Adding a second, independent source
to EVERY class breaks that correlation and is our best hedge against an unknown
test distribution (which the organizer hints is Hubble/SDSS/PDS-style).

Source: the NASA Image and Video Library public API (images-api.nasa.gov).
No key/auth. Real Hubble / spacecraft imagery, public domain, processed RGB —
the organizer's exact style. One API serves every class:
  nebula        -> "nebula" (Hubble emission/planetary nebulae)
  planet        -> Jupiter/Saturn/Mars/... (Hubble/Voyager/Cassini/Juno)
  spiral        -> "spiral galaxy"
  elliptical    -> "elliptical galaxy"
  star_cluster  -> globular + open cluster

CURATION IS THE POINT (don't bulk-dump): the library also holds illustrations,
diagrams, charts and hardware photos. We filter those out by keyword, keep only
media_type=image, drop extreme panoramas, and content-hash dedup. Then a
smoke-test grid lets you eyeball quality before the full pull.

Layout (kept separate for clean provenance; each image = one origin):
  data/nasa_lib/<class>/<class>_nasa_<nasa_id>.jpg
Integrate into data.py by treating each class as covering BOTH its original
folder(s) AND nasa_lib/<class>  (see note at bottom).

Requires: requests, pillow, matplotlib. No pip installs needed in Colab.
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
# CONFIG
# --------------------------------------------------------------------------- #
DATA_ROOT = "/content/drive/MyDrive/scale-odyssey/data"
OUT_ROOT  = os.path.join(DATA_ROOT, "nasa_lib")

API_SEARCH = "https://images-api.nasa.gov/search"
API_ASSET  = "https://images-api.nasa.gov/asset/{}"

TARGET_PER_CLASS = 100     # how many keepers to collect per class (tunable)
MAX_ASPECT       = 2.2     # reject panoramas/strips wider/taller than this
MIN_SIDE_PX      = 200     # reject tiny thumbnails
PREFERRED_SIZES  = ["~medium.jpg", "~large.jpg", "~orig.jpg", "~small.jpg"]

REQUEST_PAUSE = 0.5
RETRIES       = 3
TIMEOUT       = 60

# Per-class search plan. Multiple queries per class widen coverage; queries are
# deliberately specific to reduce junk. Order matters (best first).
CLASS_QUERIES = {
    "nebula":       ["Hubble nebula", "emission nebula", "planetary nebula"],
    "planet":       ["Jupiter planet", "Saturn planet", "Mars planet",
                     "Neptune planet", "Venus planet", "Uranus planet"],
    "spiral":       ["spiral galaxy", "barred spiral galaxy"],
    "elliptical":   ["elliptical galaxy", "lenticular galaxy"],
    "star_cluster": ["globular cluster", "open star cluster", "star cluster Hubble"],
}

# Titles/keywords that mean "not a real telescope image" -> drop.
JUNK_RX = re.compile(
    r"\b(illustration|artist|concept|rendering|render|animation|diagram|"
    r"infographic|chart|graph|map|poster|logo|model|schematic|simulation|"
    r"artwork|impression|cutaway|timeline|comparison|label(l)?ed)\b",
    re.IGNORECASE,
)


# --------------------------------------------------------------------------- #
# API helpers
# --------------------------------------------------------------------------- #
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
    """Yield (nasa_id, title) for real-image hits matching `query`, curated."""
    out, page = [], 1
    while len(out) < want and page <= 10:
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
            if not nid or not title:
                continue
            if JUNK_RX.search(blob):                       # drop illustrations etc.
                continue
            key = title.lower()
            if key in seen_titles:                          # cross-query dedup by title
                continue
            seen_titles.add(key)
            out.append((nid, title))
            if len(out) >= want:
                break
        page += 1
        time.sleep(REQUEST_PAUSE)
    return out


def asset_image_url(nasa_id):
    """Pick the best single JPG url for an asset."""
    js = _get_json(API_ASSET.format(requests.utils.quote(nasa_id)))
    if not js:
        return None
    hrefs = [i.get("href", "") for i in js.get("collection", {}).get("items", [])]
    hrefs = [h for h in hrefs if h.lower().endswith((".jpg", ".jpeg", ".png"))]
    for size in PREFERRED_SIZES:                            # prefer medium -> large -> orig
        for h in hrefs:
            if h.endswith(size):
                return h
    return hrefs[0] if hrefs else None


def fetch_image(url):
    """Download + validate one image. Returns PIL.Image(RGB) or None."""
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


# --------------------------------------------------------------------------- #
# SMOKE TEST — run FIRST, eyeball quality/relevance before the full pull.
# --------------------------------------------------------------------------- #
def smoke_test(per_class=4):
    import matplotlib.pyplot as plt
    classes = list(CLASS_QUERIES.keys())
    fig, axes = plt.subplots(len(classes), per_class,
                             figsize=(3 * per_class, 3 * len(classes)))
    for i, cls in enumerate(classes):
        seen = set()
        ids = []
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
                    ax.imshow(im)
                    ax.set_title(f"{cls}\n{ids[j][1][:24]}", fontsize=7)
            time.sleep(REQUEST_PAUSE)
    plt.tight_layout()
    plt.show()
    print("\nCheck: are these real photos of the right object? "
          "Tune CLASS_QUERIES / JUNK_RX if any class looks off, then re-run.")


# --------------------------------------------------------------------------- #
# FULL PULL
# --------------------------------------------------------------------------- #
def build(target_per_class=TARGET_PER_CLASS, only=None):
    """Fetch curated NASA-library images for each class into nasa_lib/<class>/.
    only = optional list of class names to restrict to (e.g. ['nebula'])."""
    classes = only or list(CLASS_QUERIES.keys())
    grand = {}
    for cls in classes:
        out_dir = os.path.join(OUT_ROOT, cls)
        os.makedirs(out_dir, exist_ok=True)
        manifest = os.path.join(out_dir, "manifest.csv")
        existing_hashes = set()
        # seed dedup with whatever is already there (idempotent re-runs)
        for f in os.listdir(out_dir):
            fp = os.path.join(out_dir, f)
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                try:
                    existing_hashes.add(hashlib.md5(open(fp, "rb").read()).hexdigest())
                except Exception:
                    pass

        have0 = len(existing_hashes)   # pre-existing count (fixed; hash-set grows as we save)
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
                digest = hashlib.md5(im.tobytes()).hexdigest()
                if digest in existing_hashes:               # content dedup
                    continue
                existing_hashes.add(digest)
                safe = re.sub(r"[^0-9A-Za-z_-]", "", nid)[:40] or f"img{saved}"
                fname = f"{cls}_nasa_{safe}.jpg"
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
        print(f"  {cls}: +{saved} new images -> {out_dir}")

    print("\n=== NASA-library diversification summary ===")
    for cls, n in grand.items():
        print(f"  {cls:<13} +{n}")
    print(f"Root -> {OUT_ROOT}")
    print("\nNEXT: fold nasa_lib/<class> into data.py as a SECOND source folder "
          "per class (origin = filename stem), and re-run audit_data.py.")


if __name__ == "__main__":
    smoke_test()
    # When the grid looks right:
    # build()                        # all 5 classes
    # build(only=['nebula'])         # or one class at a time
