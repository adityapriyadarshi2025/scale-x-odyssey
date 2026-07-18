"""
fetch_star_clusters.py — build the 5th class (Star Cluster) for Scale x Odyssey
==============================================================================

Strategy (see the research writeup): pull *real, color RGB* cutouts at the
coordinates of known star clusters from all-sky color cutout services. This
mirrors how the Galaxy Zoo (spiral/elliptical) class was built from SDSS
cutouts, so the style stays in-family. It resolves the earlier objection too:
we rejected GRAYSCALE FITS, not real survey data — these services return color.

Two cluster types are covered so the class spans its real morphology range:
  - globular clusters (compact, spherical, dense — resemble ellipticals)
  - open clusters (sparse, scattered bright stars — resemble irregular fields)

Cutout engines (all return color JPEG):
  - hips2fits  : all-sky, handles ANY field-of-view (resamples a HiPS). Default.
                 * PanSTARRS color HiPS  -> best quality, sky north of ~ -30 deg
                 * DSS2 color HiPS       -> all-sky fallback (south / plane)
  - legacy     : DESI Legacy Survey direct cutout (SDSS-like, matches Galaxy Zoo)
  - ps1        : Pan-STARRS fitscut color (grz), north of -30 deg

WORKFLOW
  1. Run the SMOKE TEST cell first. It fetches ~a dozen famous clusters at a
     couple of zoom levels and shows them in a grid. Eyeball: do they look like
     clusters? Is the framing right? Tune FOV_SCALE / engine, then re-run.
  2. Only when the smoke test looks good, run build_dataset() for the full pull.

Nothing about the existing 4 classes is touched. Each cluster = one origin
(one image), exactly like the Galaxy Zoo class — so no augmentation-leakage risk.

Requires: astropy, astroquery, requests, pillow, matplotlib (astroquery is the
only pip install needed in Colab).
"""

import os
import io
import csv
import time
import warnings

import requests
from PIL import Image

# --------------------------------------------------------------------------- #
# CONFIG
# --------------------------------------------------------------------------- #
DATA_ROOT   = "/content/drive/MyDrive/scale-odyssey/data"
OUT_DIR     = os.path.join(DATA_ROOT, "star_cluster")   # new class folder
MANIFEST    = os.path.join(OUT_DIR, "manifest.csv")

OUTPUT_PX   = 512          # saved image side length (px). Resize happens in transforms.
FOV_SCALE   = 3.0          # cutout FoV = FOV_SCALE x catalog cluster diameter
FOV_MIN_ARCMIN = 6.0       # never zoom in tighter than this
FOV_MAX_ARCMIN = 90.0      # cap huge open clusters (services choke past this)

ENGINE      = "hips2fits"  # "hips2fits" | "legacy" | "ps1"
PANSTARRS_HIPS = "CDS/P/PanSTARRS/DR1/color-z-zg-g"
DSS2_HIPS      = "CDS/P/DSS2/color"
DEC_SOUTH_LIMIT = -28.0    # below this, PanSTARRS/Legacy have no data -> use DSS2

REQUEST_PAUSE = 1.0        # politeness delay between fetches (seconds)
RETRIES       = 3
TIMEOUT       = 60

# --------------------------------------------------------------------------- #
# Curated famous clusters — reliable, unambiguous, great for the SMOKE TEST.
# (name, ra_deg, dec_deg, angular_diameter_arcmin). Coords are resolved live via
# Simbad when possible; these values are the fallback if resolution fails.
# --------------------------------------------------------------------------- #
CURATED = {
    "globular": [
        ("M13",      250.4235,  36.4597, 20),
        ("M5",       229.6384,   2.0810, 20),
        ("M15",      322.4930,  12.1670, 18),
        ("M3",       205.5484,  28.3773, 18),
        ("M92",      259.2808,  43.1359, 14),
        ("M22",      279.0997, -23.9048, 24),
        ("47Tuc",      6.0236, -72.0814, 30),   # south -> DSS2
        ("OmegaCen", 201.6970, -47.4795, 36),   # south -> DSS2
        ("M4",       245.8967, -26.5257, 26),
        ("M12",      251.8091,  -1.9486, 16),
    ],
    "open": [
        ("M45_Pleiades",  56.750,  24.117, 110),  # very large -> DSS2
        ("M44_Beehive",  130.100,  19.667,  95),
        ("M67",          132.825,  11.800,  25),
        ("NGC869",        34.740,  57.130,  30),
        ("NGC884",        35.580,  57.150,  30),
        ("M35",           92.270,  24.330,  28),
        ("M11_WildDuck", 282.770,  -6.270,  14),
        ("M34",           40.530,  42.770,  35),
        ("M37",           88.070,  32.550,  24),
        ("NGC6231",      253.540, -41.830,  15),  # south -> DSS2
    ],
}


# --------------------------------------------------------------------------- #
# Coordinate resolution (optional, robust)
# --------------------------------------------------------------------------- #
def resolve(name, fallback_ra, fallback_dec):
    """Prefer a live Simbad resolve; fall back to the hardcoded coordinate."""
    try:
        from astropy.coordinates import SkyCoord
        # strip our disambiguating suffixes (M45_Pleiades -> M45)
        clean = name.split("_")[0]
        c = SkyCoord.from_name(clean)
        return float(c.ra.deg), float(c.dec.deg)
    except Exception:
        return fallback_ra, fallback_dec


# --------------------------------------------------------------------------- #
# FoV logic
# --------------------------------------------------------------------------- #
def choose_fov_arcmin(diameter_arcmin):
    fov = FOV_SCALE * float(diameter_arcmin)
    return max(FOV_MIN_ARCMIN, min(FOV_MAX_ARCMIN, fov))


def pick_engine_for(dec, engine=ENGINE):
    """Auto-route far-south targets to DSS2 regardless of the chosen engine."""
    if dec < DEC_SOUTH_LIMIT and engine in ("legacy", "ps1"):
        return "hips2fits", DSS2_HIPS          # legacy/ps1 have no southern data
    if engine == "hips2fits":
        hips = PANSTARRS_HIPS if dec >= DEC_SOUTH_LIMIT else DSS2_HIPS
        return "hips2fits", hips
    return engine, None


# --------------------------------------------------------------------------- #
# Cutout URL builders
# --------------------------------------------------------------------------- #
def url_hips2fits(ra, dec, fov_arcmin, hips, px=OUTPUT_PX):
    fov_deg = fov_arcmin / 60.0
    return (
        "https://alasky.u-strasbg.fr/hips-image-services/hips2fits?"
        f"hips={requests.utils.quote(hips, safe='')}"
        f"&width={px}&height={px}&fov={fov_deg:.5f}"
        f"&projection=TAN&coordsys=icrs&ra={ra:.6f}&dec={dec:.6f}&format=jpg"
    )


def url_legacy(ra, dec, fov_arcmin, px=OUTPUT_PX):
    # pixscale (arcsec/px) chosen so px pixels span the requested FoV
    pixscale = (fov_arcmin * 60.0) / px
    size = min(px, 512)  # legacy caps cutouts around 512 px
    return (
        "https://www.legacysurvey.org/viewer/cutout.jpg?"
        f"ra={ra:.6f}&dec={dec:.6f}&layer=ls-dr9&pixscale={pixscale:.4f}"
        f"&size={size}&bands=grz"
    )


def _ps1_filetable(ra, dec, filters="grizy"):
    from astropy.table import Table
    url = f"https://ps1images.stsci.edu/cgi-bin/ps1filenames.py?ra={ra}&dec={dec}&filters={filters}"
    return Table.read(url, format="ascii")


def url_ps1_color(ra, dec, fov_arcmin, px=OUTPUT_PX):
    """Pan-STARRS fitscut color (red=z, green=i/r, blue=g)."""
    size_px = int(round((fov_arcmin * 60.0) / 0.25))   # PS1 native 0.25 "/px
    table = _ps1_filetable(ra, dec, filters="grz")
    if len(table) < 3:
        raise ValueError("PS1: <3 bands available here (likely off-footprint)")
    order = ["yzirg".find(f) for f in table["filter"]]
    table = table[list(__import__("numpy").argsort(order))]
    url = (f"https://ps1images.stsci.edu/cgi-bin/fitscut.cgi?"
           f"ra={ra}&dec={dec}&size={size_px}&format=jpg&output_size={px}")
    for param, row in zip(["red", "green", "blue"], table):
        url += f"&{param}={row['filename']}"
    return url


def build_url(engine, hips, ra, dec, fov_arcmin):
    if engine == "hips2fits":
        return url_hips2fits(ra, dec, fov_arcmin, hips)
    if engine == "legacy":
        return url_legacy(ra, dec, fov_arcmin)
    if engine == "ps1":
        return url_ps1_color(ra, dec, fov_arcmin)
    raise ValueError(f"unknown engine {engine}")


# --------------------------------------------------------------------------- #
# Fetch one image (with retries). Returns a PIL.Image or None.
# --------------------------------------------------------------------------- #
def fetch_image(engine, hips, ra, dec, fov_arcmin):
    last = None
    for attempt in range(1, RETRIES + 1):
        try:
            url = build_url(engine, hips, ra, dec, fov_arcmin)
            r = requests.get(url, timeout=TIMEOUT)
            r.raise_for_status()
            im = Image.open(io.BytesIO(r.content)).convert("RGB")
            # reject blank/failed cutouts (services sometimes return a black tile)
            extrema = im.getextrema()
            if max(hi for _, hi in extrema) < 8:
                raise ValueError("blank/near-black cutout")
            return im
        except Exception as e:
            last = e
            time.sleep(REQUEST_PAUSE * attempt)
    print(f"    fetch failed ({engine}) ra={ra:.3f} dec={dec:.3f}: {last}")
    return None


# --------------------------------------------------------------------------- #
# SMOKE TEST — run this FIRST and look at the grid before committing.
# --------------------------------------------------------------------------- #
def smoke_test(per_type=6, fov_scales=(2.0, 4.0), engine=ENGINE):
    """Fetch a few curated clusters at two zoom levels and display a grid."""
    import matplotlib.pyplot as plt

    picks = []
    for ctype in ("globular", "open"):
        picks += [(ctype, *row) for row in CURATED[ctype][:per_type]]

    ncols = len(fov_scales)
    nrows = len(picks)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 3.2 * nrows))
    if nrows == 1:
        axes = [axes]

    for i, (ctype, name, fra, fdec, diam) in enumerate(picks):
        ra, dec = resolve(name, fra, fdec)
        for j, scale in enumerate(fov_scales):
            fov = max(FOV_MIN_ARCMIN, min(FOV_MAX_ARCMIN, scale * diam))
            eng, hips = pick_engine_for(dec, engine)
            im = fetch_image(eng, hips, ra, dec, fov)
            ax = axes[i][j] if ncols > 1 else axes[i]
            if im is not None:
                ax.imshow(im)
            ax.set_title(f"{ctype[:3]} {name}\n{fov:.0f}' via {eng.split('2')[0]}",
                         fontsize=8)
            ax.axis("off")
            time.sleep(REQUEST_PAUSE)
    plt.tight_layout()
    plt.show()
    print("\nLook at framing + whether it reads as a cluster.")
    print("Adjust FOV_SCALE, FOV_MIN/MAX_ARCMIN, or ENGINE, then re-run.")


# --------------------------------------------------------------------------- #
# CATALOGS for the full pull (via VizieR / astroquery)
# --------------------------------------------------------------------------- #
def load_globulars(limit=None):
    """Harris Milky Way globular cluster catalog (VizieR VII/202). ~150 objects.
    Returns list of (name, ra_deg, dec_deg, diameter_arcmin)."""
    from astroquery.vizier import Vizier
    v = Vizier(columns=["Name", "RAJ2000", "DEJ2000", "Rad"])
    v.ROW_LIMIT = -1
    try:
        tab = v.get_catalogs("VII/202")[0]
    except Exception as e:
        print(f"globular catalog fetch failed ({e}); using curated fallback")
        return [(n, r, d, diam) for (n, r, d, diam) in CURATED["globular"]]
    out = []
    for row in tab:
        try:
            from astropy.coordinates import SkyCoord
            import astropy.units as u
            c = SkyCoord(str(row["RAJ2000"]), str(row["DEJ2000"]),
                         unit=(u.hourangle, u.deg))
            # Harris has no clean angular-size column here; assume compact ~10'
            out.append((str(row["Name"]).strip().replace(" ", ""),
                        float(c.ra.deg), float(c.dec.deg), 10.0))
        except Exception:
            continue
    return out[:limit] if limit else out


def load_open_clusters(limit=None):
    """Cantat-Gaudin+ 2020 Gaia DR2 open clusters (VizieR J/A+A/640/A1).
    Uses the 50%-radius (r50, deg) to size each cutout. ~2000 objects."""
    from astroquery.vizier import Vizier
    v = Vizier(columns=["Cluster", "RA_ICRS", "DE_ICRS", "r50"])
    v.ROW_LIMIT = -1
    try:
        tab = v.get_catalogs("J/A+A/640/A1")[0]
    except Exception as e:
        print(f"open cluster catalog fetch failed ({e}); using curated fallback")
        return [(n, r, d, diam) for (n, r, d, diam) in CURATED["open"]]
    out = []
    for row in tab:
        try:
            ra = float(row["RA_ICRS"]); dec = float(row["DE_ICRS"])
            r50_deg = float(row["r50"]) if row["r50"] else 0.05
            diam_arcmin = max(6.0, r50_deg * 60.0 * 4.0)  # ~ visible extent
            out.append((str(row["Cluster"]).strip().replace(" ", ""),
                        ra, dec, diam_arcmin))
        except Exception:
            continue
    return out[:limit] if limit else out


# --------------------------------------------------------------------------- #
# FULL DOWNLOAD
# --------------------------------------------------------------------------- #
def build_dataset(n_globular=150, n_open=250, engine=ENGINE, dry_run=False):
    """Fetch cutouts for both cluster types and save to OUT_DIR with a manifest.

    dry_run=True  -> just print the URLs it *would* fetch (no network, no writes).
    Set n_* to cap counts (balance against your other classes ~150-200 origins).
    """
    os.makedirs(OUT_DIR, exist_ok=True)
    targets = ([("gc", *t) for t in load_globulars(n_globular)] +
               [("oc", *t) for t in load_open_clusters(n_open)])
    print(f"Prepared {len(targets)} targets "
          f"({sum(1 for t in targets if t[0]=='gc')} globular, "
          f"{sum(1 for t in targets if t[0]=='oc')} open).")

    rows, ok, skip, fail = [], 0, 0, 0
    for idx, (prefix, name, ra, dec, diam) in enumerate(targets):
        safe = "".join(ch for ch in str(name) if ch.isalnum() or ch in "-_") or "unk"
        # index prefix guarantees uniqueness even if catalog names collide/repeat
        fname = f"{prefix}_{idx:04d}_{safe}.jpg"   # filename stem == origin id
        fpath = os.path.join(OUT_DIR, fname)
        fov = choose_fov_arcmin(diam)
        eng, hips = pick_engine_for(dec, engine)

        if dry_run:
            print(f"  {fname:<28} fov={fov:5.1f}'  {build_url(eng, hips, ra, dec, fov)[:90]}...")
            continue
        if os.path.exists(fpath):
            skip += 1
            continue

        im = fetch_image(eng, hips, ra, dec, fov)
        if im is None:
            fail += 1
            continue
        im.save(fpath, quality=92)
        rows.append([fname, prefix, name, f"{ra:.6f}", f"{dec:.6f}",
                     f"{fov:.2f}", eng, hips or ""])
        ok += 1
        if ok % 25 == 0:
            print(f"    ...{ok} saved")
        time.sleep(REQUEST_PAUSE)

    if dry_run:
        return

    # append/refresh manifest
    header = ["file", "type", "cluster", "ra", "dec", "fov_arcmin", "engine", "hips"]
    write_header = not os.path.exists(MANIFEST)
    with open(MANIFEST, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(header)
        w.writerows(rows)

    print(f"\nDONE. saved={ok}  skipped(existing)={skip}  failed={fail}")
    print(f"Images -> {OUT_DIR}")
    print(f"Manifest -> {MANIFEST}")
    print("Origin rule for this class = filename stem (one image per cluster), "
          "same family as the Galaxy Zoo class.")


if __name__ == "__main__":
    # Default action = the safe one: show the smoke test.
    smoke_test()
    # When happy, comment out smoke_test() above and run:
    # build_dataset(dry_run=True)     # preview URLs first
    # build_dataset()                 # real download
