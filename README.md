# Scale × Odyssey — Astronomical Image Classifier

A deep-learning system that classifies astronomical images into five celestial
object categories — **Spiral Galaxy, Elliptical Galaxy, Nebula, Star Cluster,
Planetary Object** — from raw image data, with interpretability, an
out-of-distribution study, and four bonus features. Built for Summer Siege 2026
(Technical Council).

**🚀 Live demo:** https://huggingface.co/spaces/Priyadarshi101/scale-x-odyssey
— drop in an astronomical image and the model classifies it **entirely in your
browser** (ONNX / onnxruntime-web, no server), with confidence bars, a caption,
and an out-of-distribution flag.

## Results at a glance

- **94–95 % test accuracy** on a held-out, leakage-safe split — clears the ~93 %
  target. (Baseline model 95.2 %; diversified model 93.8 % on a harder, balanced
  test set with markedly stronger galaxy classes.)
- **Real-Hubble out-of-distribution (OOD) test** — an honest generalisation probe
  on 286 ESA/Hubble images the model never trained on.
- **Grad-CAM interpretability** — attention maps confirm the model looks at the
  object, not the background.
- **All four Phase-4 bonus features**: interactive web demo, anomaly detection,
  object localization, and image captioning.

Full write-up with all numbers and analysis: **[`docs/RESULTS.md`](docs/RESULTS.md)**.

## Key finding

A controlled data-diversification experiment produced a clean, honest result:
expanding the weak galaxy classes with a *new survey* (DECaLS) markedly raised
their in-distribution accuracy (elliptical F1 **0.81 → 0.90**), yet OOD
generalisation improved only for the class given genuinely *cross-instrument*
variety (planet, OOD F1 0.62 → 0.69). The takeaway: **out-of-distribution gains
require instrument-diverse data, not merely more data.**

## Model & method

EfficientNet-B0 pretrained on ImageNet, fine-tuned end-to-end with a fresh
5-class head (~4 M params). Geometry-heavy / colour-light augmentation (rotations
and flips are free for astronomical objects; colour is kept faithful).
AdamW + cosine schedule, class-weighted loss, best-validation checkpointing.
Everything is config-driven (`configs/default.yaml`) and seeded.

## Dataset

Multi-source (2–3 independent real sources per class) to prevent single-source
overfitting — 5,851 images across the five classes, plus a separate Hubble OOD
test set. Sources: SpaceNet FLARE, Galaxy Zoo 2 (SDSS), DESI Legacy Survey
(DECaLS / Galaxy10), Pan-STARRS/DSS2 survey cutouts, the NASA Image Library, and
ESA/Hubble (OOD). Split is **leakage-safe on origins** (augmented variants of one
object never straddle train/test). Acquisition is fully scripted in `scripts/`.

## Repository structure

```
src/
  data.py         leakage-safe file index + origin split + AstroDataset
  transforms.py   train vs val/test transforms
  model.py        EfficientNet-B0 backbone + 5-class head
  train.py        training loop (class-weighted, cosine, early stop)
  eval.py         metrics, confusion matrix, Grad-CAM, OOD evaluation
  anomaly.py      bonus: out-of-distribution / anomaly detection
  localize.py     bonus: Grad-CAM weakly-supervised bounding boxes
  caption.py      bonus: natural-language image captions
app.py            bonus: interactive Gradio web demo
scripts/          one-off data acquisition (fetchers + audit)
configs/          paths + hyperparameters
docs/RESULTS.md   full results report
notebooks/        Colab driver
```

Image data and model checkpoints live on Google Drive (too large for git);
the code clones from here and mounts Drive at run time.

## Quickstart (Colab)

```bash
git clone https://github.com/adityapriyadarshi2025/scale-x-odyssey.git
pip install -r scale-x-odyssey/requirements.txt
```

```python
# train
python -m src.train  --data_root <data>
# evaluate (test metrics + confusion + Grad-CAM)
python -m src.eval   --data_root <data> --checkpoint <best_model.pth>
```

## Bonus features

- **Web demo** — `python app.py` → public link; upload an image for class +
  confidence + Grad-CAM overlay.
- **Anomaly detection** — `python -m src.anomaly …` flags images unlike the 5
  classes (max-softmax / energy); AUROC ≈ 0.76 vs the Hubble OOD set.
- **Object localization** — `python -m src.localize …` draws bounding boxes from
  Grad-CAM, no box labels.
- **Image captioning** — `python -m src.caption …` writes descriptions like
  *"A spiral galaxy with a bright central core and extended spiral arms."*
