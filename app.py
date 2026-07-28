"""
app.py — interactive web demo for Scale x Odyssey (Phase 4 bonus).

Drag in an astronomical image; the model predicts one of the 5 classes, shows
per-class confidences, and overlays a Grad-CAM map of the pixels it used. Reuses
src/model.py, src/transforms.py, src/eval.py.

Run from the repo root:
    pip install gradio
    python app.py                 # prints a public share link (works in Colab)
"""

import os
import yaml
import numpy as np
import torch
import gradio as gr
from PIL import Image

from src import model as M
from src import transforms as T
from src.data import LABELS
from src.eval import _target_layer

# --------------------------------------------------------------------------- #
CFG = yaml.safe_load(open("configs/default.yaml"))
CLASSES = CFG["classes"]                       # index order matches LABELS
SIZE = CFG["data"]["image_size"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CKPT_CANDIDATES = [
    "/content/artifacts/best_model.pth",
    "/content/drive/MyDrive/scale-odyssey/artifacts/best_model.pth",
    "artifacts/best_model.pth",
]


def _load_model():
    ckpt = next((p for p in CKPT_CANDIDATES if os.path.isfile(p)), None)
    if ckpt is None:
        raise FileNotFoundError(f"no checkpoint found in {CKPT_CANDIDATES}")
    model = M.from_config(CFG)
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE)["model"])
    print(f"loaded {ckpt}")
    return model.to(DEVICE).eval()


MODEL = _load_model()
EVAL_TF = T.eval_transforms(SIZE, CFG["data"]["norm_mean"], CFG["data"]["norm_std"])

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
CAM = GradCAM(model=MODEL, target_layers=[_target_layer(MODEL)])


def predict(pil_img):
    if pil_img is None:
        return {}, None
    img = pil_img.convert("RGB")
    x = EVAL_TF(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        probs = torch.softmax(MODEL(x), dim=1)[0].cpu().numpy()
    confidences = {CLASSES[i]: float(probs[i]) for i in range(len(CLASSES))}

    heat = CAM(input_tensor=x)[0]                         # HxW in [0,1]
    rgb = np.array(img.resize((SIZE, SIZE))).astype(np.float32) / 255.0
    overlay = show_cam_on_image(rgb, heat, use_rgb=True)
    return confidences, Image.fromarray(overlay)


DESCRIPTION = (
    "Upload an astronomical image. The model classifies it as one of "
    "**Spiral Galaxy, Elliptical Galaxy, Nebula, Star Cluster, or Planetary "
    "Object**, shows its confidence in each class, and highlights (Grad-CAM) the "
    "regions it used to decide."
)

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Astronomical image"),
    outputs=[
        gr.Label(num_top_classes=5, label="Prediction & confidence"),
        gr.Image(label="Grad-CAM — where the model looked"),
    ],
    title="Scale × Odyssey — Astronomical Image Classifier",
    description=DESCRIPTION,
    allow_flagging="never",
)

if __name__ == "__main__":
    demo.launch(share=True)
