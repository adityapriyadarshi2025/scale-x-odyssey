"""
transforms.py — image transforms for Scale x Odyssey

Two pipelines:
  * TRAIN: resize/crop -> geometry augmentation -> (light photometric) -> normalize
  * EVAL  (val/test/ood): resize -> center-crop -> normalize   (NO augmentation)

Design choice — geometry-heavy, and (revised) colour-ROBUST:
  Astronomical objects have NO canonical orientation or handedness. A galaxy is
  just as valid rotated 37 deg or mirrored, so full rotations + h/v flips are
  "free" label-preserving augmentation and the strongest lever we have.
  COLOUR: the original pipeline kept photometric jitter minimal (brightness/
  contrast only, no hue/saturation) to preserve physically-meaningful colour.
  A diagnostic on real Hubble imagery showed this backfired: the model became
  colour-BRITTLE — desaturating a Hubble spiral swung its prediction across three
  classes (star_cluster -> planet), proving it keyed on colour statistics rather
  than morphology, which is why cross-instrument (Hubble-processed) images fail.
  The pipeline now deliberately adds hue/saturation jitter and occasional random
  grayscale so the network cannot lean on one instrument's exact colour palette
  and is forced onto shape. A little blur/noise mimics different telescopes'
  seeing/resolution. Trade-off: colour still carries real signal for nebula/
  planet, so jitter is moderate and the effect is measured before/after on both
  the in-distribution and the Hubble OOD test.

Usage:
    from src.transforms import get_transforms
    from src.data import build_file_index, split_on_origins, AstroDataset
    train_tf, eval_tf = get_transforms(image_size=224)
    splits = split_on_origins(build_file_index())
    train_ds = AstroDataset(splits["train"], transform=train_tf)
    val_ds   = AstroDataset(splits["val"],   transform=eval_tf)
"""

from torchvision import transforms

# ImageNet stats — match the pretrained backbone (EfficientNet-B0 etc.)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def train_transforms(image_size=224, mean=IMAGENET_MEAN, std=IMAGENET_STD,
                     strength="standard"):
    """Augmentation pipeline for training.

    strength: "light" | "standard" | "heavy" — dials geometry/photometric intensity.
    Black (fill=0) is used for rotation borders because space is black, so rotated
    corners blend in instead of introducing grey edges.
    """
    if strength not in ("light", "standard", "heavy"):
        raise ValueError("strength must be light|standard|heavy")

    # scale range for the random zoom/crop (gentler when lighter)
    scale = {"light": (0.9, 1.0), "standard": (0.8, 1.0), "heavy": (0.65, 1.0)}[strength]
    # brightness/contrast jitter magnitude
    bc = {"light": 0.0, "standard": 0.15, "heavy": 0.25}[strength]
    # colour-ROBUSTNESS: saturation + hue jitter (breaks reliance on exact palette)
    sat = {"light": 0.0, "standard": 0.30, "heavy": 0.50}[strength]
    hue = {"light": 0.0, "standard": 0.05, "heavy": 0.10}[strength]
    # probability of dropping colour entirely (forces the net onto shape)
    gray_p = {"light": 0.0, "standard": 0.15, "heavy": 0.25}[strength]

    ops = [
        # random zoom + reframe to the target size (keeps object roughly centred)
        transforms.RandomResizedCrop(image_size, scale=scale, ratio=(0.9, 1.1)),
        # orientation is meaningless -> free symmetry augmentation
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(180, fill=0),
    ]

    # photometric jitter — now includes saturation + hue for colour robustness
    if bc > 0 or sat > 0 or hue > 0:
        ops.append(transforms.ColorJitter(brightness=bc, contrast=bc,
                                          saturation=sat, hue=hue))
    # occasionally drop colour so the model must decide on morphology
    if gray_p > 0:
        ops.append(transforms.RandomGrayscale(p=gray_p))

    # mild blur to mimic different telescope seeing/resolution (cross-instrument)
    if strength in ("standard", "heavy"):
        ops.append(transforms.RandomApply(
            [transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5))], p=0.2))

    ops += [
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ]
    return transforms.Compose(ops)


def eval_transforms(image_size=224, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    """Deterministic pipeline for val / test / ood — resize, center-crop, normalize.
    Resize on the shorter side preserves aspect ratio; center-crop makes it square."""
    return transforms.Compose([
        transforms.Resize(image_size),          # shorter side -> image_size (keeps aspect)
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


def get_transforms(image_size=224, mean=IMAGENET_MEAN, std=IMAGENET_STD,
                   strength="standard"):
    """Convenience: return (train_tf, eval_tf)."""
    return (train_transforms(image_size, mean, std, strength),
            eval_transforms(image_size, mean, std))


if __name__ == "__main__":
    # quick self-test: build both pipelines and run a dummy RGB image through
    from PIL import Image
    tr, ev = get_transforms(224)
    dummy = Image.new("RGB", (640, 480), (20, 40, 80))
    xt = tr(dummy)
    xe = ev(dummy)
    print("train tensor:", tuple(xt.shape), "dtype", xt.dtype,
          "range [%.2f, %.2f]" % (xt.min(), xt.max()))
    print("eval  tensor:", tuple(xe.shape))
    assert xt.shape == (3, 224, 224) and xe.shape == (3, 224, 224)
    print("OK — both pipelines output 3x224x224 tensors.")
