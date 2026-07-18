# Scale × Odyssey

A 5-class deep-learning classifier for astronomical images — **Spiral Galaxy,
Elliptical Galaxy, Nebula, Star Cluster, Planetary Object** — built for Summer
Siege 2026 (Technical Council).

Baseline to beat: EfficientNet-B0 ≈ 93% on the 5-class split.
Evaluation: in-distribution accuracy **plus** a held-out out-of-distribution
(OOD) test on real telescope imagery.

---

## Where things live

Code and data are deliberately kept apart:

| | Location | In git? |
|---|---|---|
| **Code** (`src/`, `scripts/`, `configs/`) | this GitHub repo | ✅ yes |
| **Data** (images) | Google Drive `MyDrive/scale-odyssey/data/` | ❌ no (too big) |
| **Artifacts** (`split.json`, checkpoints) | Google Drive `.../artifacts/` | split.json yes, checkpoints no |

Colab is only a **driver**: it mounts Drive (data) and clones this repo (code),
then imports from `src/`. Nothing important lives in the notebook itself, so a
runtime reset never loses work.

## Repo layout

```
src/
  data.py          leakage-safe file index + origin split + AstroDataset
  transforms.py    train vs val/test transforms          (to build)
  model.py         backbone + classification head         (to build)
  train.py         training loop                          (to build)
  eval.py          metrics, Grad-CAM, OOD evaluation       (to build)
scripts/
  fetch_star_clusters.py   builds the star_cluster class (survey cutouts)
  fetch_nasa_library.py    second-source diversification (NASA image library)
  audit_data.py            data-quality audit across all classes
configs/
  default.yaml     paths + hyperparameters (nothing hardcoded)
notebooks/
  driver.ipynb     the Colab driver (mount + clone + run)
```

## Dataset

Five classes, each drawn from **two independent sources** (to prevent a model
from keying on a single source's fingerprint):

| Class | Primary source | Second source |
|---|---|---|
| nebula | SpaceNet FLARE | NASA Image Library |
| planet | SpaceNet FLARE | NASA Image Library |
| spiral | Galaxy Zoo 2 | NASA Image Library |
| elliptical | Galaxy Zoo 2 | NASA Image Library |
| star_cluster | Pan-STARRS/DSS2 cutouts | NASA Image Library |

**Leakage rule:** the train/val/test split is on *origins*, not files — all
augmented variants of one original stay on the same side. Enforced by a hard
assertion in `data.py`.

## Quickstart (Colab)

1. Open `notebooks/driver.ipynb` in Colab.
2. Run the setup cells: mount Drive, clone this repo, `pip install -r requirements.txt`.
3. Build the split:
   ```python
   from src import data
   data.prepare_splits()                          # normal split
   # data.prepare_splits(hold_out_source="nasa_lib")   # reserve NASA as OOD test
   ```
4. (Next) train: `python -m src.train --config configs/default.yaml`

## Dev workflow

Build/iterate in Colab → download the `.py` → commit & push from your machine →
Colab `git pull` to pick it up. Data-acquisition scripts in `scripts/` are
one-offs; they've already been run and the images are on Drive.
