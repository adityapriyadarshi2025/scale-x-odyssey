"""
audit_data.py — Scale x Odyssey data-quality audit
===================================================

Run this in Colab AFTER mounting Drive. It inspects every image already
downloaded to /content/drive/MyDrive/scale-odyssey/data/ and reports:

  1. Per-class file counts and unique-origin counts.
  2. Origin variant distribution (min/max/mean variants per origin, thin origins).
  3. Cross-class contamination (a file whose true class prefix != its folder).
  4. Duplicate detection — by filename AND by content hash (the real dedup).
  5. Image integrity — corrupt/unreadable files, plus mode / size / channel stats.
  6. Colour-mode split (RGB vs grayscale) — matters for the SpaceNet/GalaxyZoo
     style clash and for the future star-cluster class.

Nothing is modified. This is read-only inspect-before-committing. A JSON report
is written to the data root so the numbers are reproducible and judge-inspectable.

Why this exists: a Drive-level count already found duplicate-TITLED files
(28 in nebula, 34 in planet) that inflate counts above the report's 1152/1480.
Google Drive allows same-name files in one folder; the FUSE mount can expose
them inconsistently between runs, so we verify against CONTENT HASHES here,
not filenames.
"""

import os
import re
import csv
import json
import hashlib
from collections import defaultdict, Counter
from datetime import datetime

from PIL import Image
Image.MAX_IMAGE_PIXELS = None  # these are SwinIR-upscaled; don't trip the bomb guard

# --------------------------------------------------------------------------- #
# Config — one entry per class folder, with its origin-extraction rule.
# --------------------------------------------------------------------------- #
DATA_ROOT = "/content/drive/MyDrive/scale-odyssey/data"   # default; auto-corrected below

# --- robustness knobs ------------------------------------------------------ #
# Reading 3000+ full images directly over the Drive FUSE mount is what caused
# the "Transport endpoint is not connected" crash. Stage to local disk first.
STAGE_LOCALLY = True                       # copy Drive -> local Colab disk, audit local
LOCAL_ROOT    = "/content/so_data"         # where the local copy lives
FORCE_REMOUNT = True                       # cleanly remount before touching Drive
HASH_CHECK    = True                       # content-hash dedup (cheap on local disk)


def _remount_drive():
    """Cleanly remount Drive. Fixes a mount that has already disconnected."""
    try:
        from google.colab import drive
        if FORCE_REMOUNT:
            try:
                drive.flush_and_unmount()
            except Exception:
                pass
            print("Remounting Drive (force)...")
            drive.mount("/content/drive", force_remount=True)
        elif not os.path.isdir("/content/drive"):
            drive.mount("/content/drive")
    except Exception as e:
        print(f"(Not in Colab or mount failed: {e})")


def _stage_to_local(src_root):
    """Copy the class subfolders from Drive to local disk. One sustained copy is
    far more reliable than thousands of individual reads over FUSE. Returns the
    local root to audit (falls back to src_root if staging fails)."""
    import shutil
    os.makedirs(LOCAL_ROOT, exist_ok=True)
    print(f"Staging data to local disk: {src_root} -> {LOCAL_ROOT}")
    for cfg in CLASSES.values():
        sub = cfg["subpath"]
        src = os.path.join(src_root, sub)
        dst = os.path.join(LOCAL_ROOT, sub)
        if not os.path.isdir(src):
            print(f"  skip (not found on Drive): {sub}")
            continue
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        # retry the copy a couple of times in case FUSE hiccups mid-way
        for attempt in range(1, 4):
            try:
                shutil.copytree(src, dst, dirs_exist_ok=True)
                n = len(os.listdir(dst))
                print(f"  {sub}: {n} files staged")
                break
            except Exception as e:
                print(f"  {sub}: copy attempt {attempt} failed ({e}); retrying...")
                _remount_drive()
        else:
            print(f"  {sub}: STILL failing after retries — will audit from Drive.")
            return src_root
    return LOCAL_ROOT


def _ensure_drive_and_locate():
    """Mount Drive if needed, then find the real scale-odyssey/data path.

    'FOLDER MISSING' almost always means Drive isn't mounted in this runtime,
    or the folder lives somewhere other than MyDrive root (e.g. a Shared drive).
    This resolves both without guessing.
    """
    global DATA_ROOT
    import glob

    # 1. Cleanly (re)mount — a previously-disconnected FUSE endpoint is the
    #    usual cause of files reading as missing/corrupt.
    _remount_drive()

    # 2. If the default path works, keep it.
    if os.path.isdir(DATA_ROOT):
        return DATA_ROOT

    # 3. Otherwise search likely roots for a 'scale-odyssey/data' folder.
    print(f"Default path not found ({DATA_ROOT}); searching for it...")
    search_bases = [
        "/content/drive/MyDrive",
        "/content/drive/My Drive",
        "/content/drive/Shareddrives",
        "/content/drive/Shared drives",
    ]
    hits = []
    for base in search_bases:
        if os.path.isdir(base):
            hits += glob.glob(f"{base}/**/scale-odyssey/data", recursive=True)
            hits += glob.glob(f"{base}/**/scale-odyssey", recursive=True)  # data/ maybe nested differently
    hits = sorted(set(hits), key=len)
    for h in hits:
        cand = h if os.path.basename(h) == "data" else os.path.join(h, "data")
        if os.path.isdir(os.path.join(cand, "spacenet_raw")) or \
           os.path.isdir(os.path.join(cand, "galaxy_zoo")):
            DATA_ROOT = cand
            print(f"Located data root: {DATA_ROOT}")
            return DATA_ROOT

    print("!! Could not locate scale-odyssey/data anywhere under /content/drive.")
    print("   Mounted contents of /content/drive:")
    for p in glob.glob("/content/drive/*"):
        print("    ", p)
    return DATA_ROOT

# SpaceNet variant suffix: base, _<0-4>, or _augN, before _SwinIR_large.png
SPACENET_RX = re.compile(r"(_(?:[0-4]|aug\d+))?_SwinIR_large\.png$")

def spacenet_origin(fname):
    """nebula_page_6_image_20_aug3_SwinIR_large.png -> nebula_page_6_image_20"""
    return SPACENET_RX.sub("", fname)

def galaxyzoo_origin(fname):
    """13178.jpg -> 13178  (each Galaxy Zoo file is its own origin)"""
    return os.path.splitext(fname)[0]

# subpath is relative to DATA_ROOT; the absolute path is resolved in main()
CLASSES = {
    "nebula":     {"subpath": "spacenet_raw/nebula",   "origin": spacenet_origin,  "prefix": "nebula"},
    "planet":     {"subpath": "spacenet_raw/planet",   "origin": spacenet_origin,  "prefix": "planet"},
    "spiral":     {"subpath": "galaxy_zoo/spiral",     "origin": galaxyzoo_origin, "prefix": None},
    "elliptical": {"subpath": "galaxy_zoo/elliptical", "origin": galaxyzoo_origin, "prefix": None},
    # 5th class: one image per cluster, origin = filename stem (gc_M13 / oc_M67)
    "star_cluster": {"subpath": "star_cluster",        "origin": galaxyzoo_origin, "prefix": None},
}

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def md5_of(path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def audit_class(name, cfg):
    path = os.path.join(DATA_ROOT, cfg["subpath"])
    origin_fn, prefix = cfg["origin"], cfg["prefix"]
    report = {"class": name, "path": path}

    if not os.path.isdir(path):
        report["error"] = "FOLDER MISSING"
        return report

    files = [f for f in os.listdir(path)
             if os.path.splitext(f)[1].lower() in IMG_EXTS]
    report["file_count"] = len(files)

    origins = defaultdict(list)          # origin_id -> [filenames]
    contamination = []                   # files whose prefix != folder class
    unparsed = []                        # names the origin rule couldn't handle
    hash_map = defaultdict(list)         # content md5 -> [filenames]
    corrupt = []                         # (filename, error)
    modes, sizes, channels = Counter(), Counter(), Counter()

    for f in files:
        fp = os.path.join(path, f)

        # origin + contamination
        oid = origin_fn(f)
        origins[oid].append(f)
        if prefix and not f.startswith(prefix + "_"):
            contamination.append(f)
        if prefix and not SPACENET_RX.search(f):
            unparsed.append(f)

        # content hash (true duplicate detection)
        if HASH_CHECK:
            try:
                hash_map[md5_of(fp)].append(f)
            except Exception as e:
                corrupt.append((f, f"hash: {e}"))
                continue

        # integrity + properties
        try:
            with Image.open(fp) as im:
                im.verify()                       # catches truncation/corruption
            with Image.open(fp) as im:            # reopen after verify
                modes[im.mode] += 1
                sizes[im.size] += 1
                channels[len(im.getbands())] += 1
        except Exception as e:
            corrupt.append((f, f"open: {e}"))

    variant_counts = {o: len(v) for o, v in origins.items()}
    vals = list(variant_counts.values()) or [0]
    dup_hashes = {h: fs for h, fs in hash_map.items() if len(fs) > 1}
    dup_extra_files = sum(len(fs) - 1 for fs in dup_hashes.values())

    report.update({
        "unique_origins": len(origins),
        "variants_per_origin": {
            "min": min(vals), "max": max(vals),
            "mean": round(sum(vals) / len(vals), 2),
        },
        "thin_origins_le2": sorted(o for o, c in variant_counts.items() if c <= 2),
        "contamination_files": contamination,
        "unparsed_names": unparsed,
        "corrupt_files": corrupt,
        "duplicate_content_groups": len(dup_hashes),
        "duplicate_extra_files": dup_extra_files,
        "duplicate_examples": dict(list(dup_hashes.items())[:5]),
        "modes": dict(modes),
        "channels": dict(channels),
        "distinct_sizes": len(sizes),
        "top_sizes": sizes.most_common(5),
    })
    return report


def main():
    global DATA_ROOT
    drive_root = _ensure_drive_and_locate()      # the real Drive path
    if STAGE_LOCALLY:
        DATA_ROOT = _stage_to_local(drive_root)  # audit the fast, stable local copy
    print("=" * 70)
    print("Scale x Odyssey — DATA AUDIT")
    print(f"auditing from: {DATA_ROOT}")
    print(datetime.now().isoformat(timespec="seconds"))
    print("=" * 70)

    full = {"generated_at": datetime.now().isoformat(), "classes": {}}
    grand_files = grand_origins = 0

    for name, cfg in CLASSES.items():
        r = audit_class(name, cfg)
        full["classes"][name] = r
        print(f"\n### {name.upper()}  ({r.get('path')})")
        if r.get("error"):
            print(f"  !! {r['error']}")
            continue
        grand_files += r["file_count"]
        grand_origins += r["unique_origins"]
        vpo = r["variants_per_origin"]
        print(f"  files ................ {r['file_count']}")
        print(f"  unique origins ....... {r['unique_origins']}")
        print(f"  variants/origin ...... min {vpo['min']}  max {vpo['max']}  mean {vpo['mean']}")
        print(f"  thin origins (<=2) ... {len(r['thin_origins_le2'])}")
        print(f"  contamination ........ {len(r['contamination_files'])}"
              f"  {r['contamination_files'][:5]}")
        print(f"  unparsed names ....... {len(r['unparsed_names'])}"
              f"  {r['unparsed_names'][:5]}")
        print(f"  CORRUPT / unreadable . {len(r['corrupt_files'])}"
              f"  {[c[0] for c in r['corrupt_files'][:5]]}")
        print(f"  duplicate content .... {r['duplicate_content_groups']} groups,"
              f" {r['duplicate_extra_files']} redundant files")
        print(f"  colour modes ......... {r['modes']}   channels {r['channels']}")
        print(f"  distinct sizes ....... {r['distinct_sizes']}   top {r['top_sizes'][:3]}")

    print("\n" + "=" * 70)
    print(f"TOTAL usable images (as files) : {grand_files}")
    print(f"TOTAL unique origins           : {grand_origins}")
    print("Class balance (by origin):")
    for name, r in full["classes"].items():
        if "unique_origins" in r:
            bar = "#" * max(1, r["unique_origins"] // 10)
            print(f"  {name:<11} {r['unique_origins']:>4}  {bar}")
    print("=" * 70)

    # persist reproducible report — always to local disk, then best-effort to Drive
    local_base = "/content" if os.path.isdir("/content") else os.getcwd()
    local_out = os.path.join(local_base, "audit_report.json")
    with open(local_out, "w") as f:
        json.dump(full, f, indent=2)
    print(f"\nFull JSON report written to: {local_out}")
    try:
        import shutil
        drive_out = os.path.join(drive_root, "audit_report.json")
        shutil.copy(local_out, drive_out)
        print(f"Also copied to Drive: {drive_out}")
    except Exception as e:
        print(f"(Could not copy report to Drive — download {local_out} manually: {e})")

    return full


if __name__ == "__main__":
    main()
