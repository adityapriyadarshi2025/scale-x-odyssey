# Scale × Odyssey — Results Report

**Task.** Classify astronomical images into five celestial-object categories —
Spiral Galaxy, Elliptical Galaxy, Nebula, Star Cluster, Planetary Object — from
raw image data alone, without hand-engineered astrophysical features.

**Headline result.** A fine-tuned EfficientNet-B0 reaches **95.2 % accuracy on a
held-out test set** (96.4 % on validation), clearing the project's baseline
target of ~93 %. The evaluation is leakage-safe and reproducible from a fixed
seed, and is accompanied by per-class metrics, a confusion matrix, and Grad-CAM
attention maps.

---

## 1. Dataset

The dataset is assembled from multiple public sources, with **every class drawn
from two independent sources**. This is deliberate: a single-source class lets a
network key on a source's "fingerprint" (colour calibration, noise, resolution)
instead of the object itself, which then collapses on unfamiliar test imagery.
A second, independent source per class breaks that shortcut and is the main
hedge against an unknown test distribution.

| Class | Primary source | Second source | Files | Origins |
|---|---|---|---|---|
| nebula | SpaceNet FLARE | NASA Image Library | 1,202 | 194 |
| planet | SpaceNet FLARE | NASA Image Library | 1,530 | 235 |
| spiral | Galaxy Zoo 2 | NASA Image Library | 250 | 250 |
| elliptical | Galaxy Zoo 2 | NASA Image Library | 250 | 250 |
| star_cluster | Pan-STARRS / DSS2 cutouts | NASA Image Library | 447 | 447 |
| **Total** | | | **3,679** | **1,376** |

Source notes. The **SpaceNet FLARE** nebula/planet images are SwinIR-upscaled
2048×2048 RGB, with roughly eight augmented variants per original. **Galaxy
Zoo 2** spiral/elliptical images are SDSS colour cutouts selected from the
Hart16 debiased morphology catalogue at a ≥0.90 confidence threshold. The
**star-cluster** class was built by fetching colour cutouts from all-sky survey
services (Pan-STARRS where covered, DSS2 elsewhere) at the coordinates of
catalogued clusters — globular clusters from the Harris catalogue and open
clusters from the Cantat-Gaudin Gaia DR2 catalogue. This mirrors how the Galaxy
Zoo class was itself produced (survey cutouts at object coordinates), keeping the
class stylistically in-family. The **NASA Image and Video Library** supplies the
curated second source for all five classes, filtered to real photographs
(illustrations, diagrams and hardware shots removed).

### Leakage-safe splitting

The train/validation/test split (70/15/15, seed 42) is performed on **origins,
not files**. An "origin" is a single original object; all augmented variants of
one original share an origin id and are forced onto the same side of the split.
A hard assertion aborts the pipeline if any origin ever appears in two splits.
The split is stratified per class so class balance is preserved across the three
partitions (2,550 train / 562 val / 567 test).

During development the origin-extraction rule for SpaceNet filenames was found to
mis-handle single-digit image numbers — it stripped the image number itself,
splitting an original's base image from its own augmented variants across the
train/test boundary. This latent leakage was corrected before any split was run
on real data.

### Data quality

An automated audit of every image confirmed the dataset is clean: no corrupt or
unreadable files, no cross-class contamination, and only a handful of
byte-identical duplicates (detected by content hash, not filename). All images
are RGB.

---

## 2. Methodology

**Model — transfer learning.** The classifier is an **EfficientNet-B0** backbone
pretrained on ImageNet, with its original 1000-class head replaced by a fresh
5-class head (~4.0 M trainable parameters). The whole network is fine-tuned (not
frozen) at a low learning rate: astronomical images differ enough from ImageNet
that letting the convolutional features adapt is worthwhile, and the dataset size
plus augmentation make full fine-tuning safe. EfficientNet-B0 is chosen because
it is the project's stated baseline architecture, and because its accuracy-per-
parameter makes training fast and inference comfortably under the 5-second
requirement.

**Augmentation — geometry-heavy, colour-light.** Training images receive a
random resized crop, horizontal and vertical flips, and full 180° rotations —
all label-preserving, since astronomical objects have no canonical orientation
or handedness. Photometric jitter is kept minimal (mild brightness/contrast
only, no hue/saturation shifts), because colour can carry physical meaning. A
light, occasional Gaussian blur is included deliberately to mimic different
telescopes' seeing and resolution, adding cross-instrument robustness.
Validation and test images are only resized, centre-cropped and normalised.

**Training.** AdamW (lr 3×10⁻⁴, weight decay 10⁻⁴) with a cosine-annealing
schedule; cross-entropy loss with **class weighting** (the galaxy classes have
far fewer files than nebula/planet) and 0.1 label smoothing; batch size 32. The
model is validated every epoch, the best-validation checkpoint is kept, and
training early-stops after six epochs without validation improvement. The run
converged and stopped at epoch 24, with the best checkpoint from epoch 18.
Everything is config-driven and seeded for reproducibility.

---

## 3. Results

Best validation accuracy was **0.964**; held-out **test accuracy 0.952**. The
small val-to-test gap indicates the model generalises rather than memorising.

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| planet | 0.996 | 0.980 | 0.988 | 246 |
| nebula | 0.972 | 0.977 | 0.975 | 177 |
| spiral | 0.895 | 0.895 | 0.895 | 38 |
| star_cluster | 0.882 | 0.882 | 0.882 | 68 |
| elliptical | 0.780 | 0.842 | 0.810 | 38 |
| **accuracy** | | | **0.952** | 567 |
| macro avg | 0.905 | 0.915 | 0.910 | 567 |
| weighted avg | 0.954 | 0.952 | 0.953 | 567 |

A confusion matrix (`confusion_test.png`) and Grad-CAM attention maps
(`gradcam.png`) accompany these numbers. The Grad-CAM overlays confirm the model
attends to the celestial object itself rather than background artefacts,
satisfying the interpretability requirement.

---

## 4. Analysis

Nebula and planet are classified almost perfectly (F1 0.975 and 0.988). They have
by far the most data and are the most visually distinct classes, so this is
expected.

The **galaxy classes are the weak point, elliptical worst (F1 0.810)**. Two
factors explain it. First, morphology: elliptical galaxies are smooth,
featureless ovals that are genuinely easy to confuse with the rounder star
clusters and, to a lesser extent, with spirals. Second, and more fundamentally,
data volume: spiral and elliptical have only ~200 origins each (38 test images
apiece) versus 1,200+ for nebula/planet, so the model both saw the fewest
examples and is evaluated on the noisiest, smallest per-class samples. Class
weighting mitigates but cannot manufacture diversity that is not in the data.

This is an actionable, evidence-driven finding: the highest-value next step for
raising the headline number is to **increase the spiral and elliptical training
data** (e.g. additional Galaxy Zoo images at the same confidence threshold),
rather than further tuning the model.

---

## 5. Reproducibility & engineering

Code lives in a GitHub repository (`src/` pipeline modules, `scripts/` for
one-off data acquisition, `configs/default.yaml` for all paths and
hyperparameters); image data and artifacts live on Google Drive. Training runs
in Colab, which clones the repo and mounts Drive. The pipeline is fully
config-driven and seeded, and the train/val/test split is persisted to JSON so
results are exactly reproducible and inspectable.

---

## 6. Open items & next steps

- **Boost spiral/elliptical data** — the model's identified bottleneck; the
  single most valuable improvement.
- **Out-of-distribution evaluation** — re-run with the NASA-library imagery held
  out as a separate OOD test to quantify cross-instrument generalisation.
- **Hyperparameter / backbone scaling** — B1–B3 for additional ceiling once the
  data is balanced.
- **Bonus tasks (optional rubric)** — image captioning, object localisation,
  anomaly detection, interactive web demo.
- **Confirm with organisers** that using real survey imagery for the
  star-cluster class is acceptable (it aligns with the referenced SDSS/Hubble
  sources, but was a substituted source).
