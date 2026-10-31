#!/usr/bin/env python3
"""Diabetic Retinopathy Detector: Flask + PyTorch (EfficientNet-B0) + LIME."""
import io
import logging
import math
import os
import uuid

import numpy as np
import torch
import torch.nn as nn
from flask import Flask, jsonify, render_template, request
from lime import lime_image
from PIL import Image
from skimage.segmentation import mark_boundaries
from torchvision import models, transforms

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("dr_app")

# --- Settings (can be changed with environment variables) ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
IMG_SIZE = 224
THRESHOLD = 0.5                                   # P(No_DR) >= THRESHOLD -> "No_DR"
LIME_SAMPLES = int(os.getenv("LIME_SAMPLES", "500"))
MAX_OUTPUT_FILES = 50                             # old LIME images beyond this are deleted
CLASS_NAMES = ["DR", "No_DR"]                     # index 0 = DR, index 1 = No_DR

app = Flask(__name__, static_folder="static", template_folder="templates")
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024   # reject uploads over 10 MB
Image.MAX_IMAGE_PIXELS = 50_000_000                    # guard against decompression bombs
OUTPUT_DIR = os.path.join(app.static_folder, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Preprocessing must match what you used in training.
test_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# --- Model loading ---
def build_model():
    m = models.efficientnet_b0(weights=None)
    m.classifier[1] = nn.Linear(1280, 1)  # one logit: sigmoid -> P(No_DR)
    return m


def find_checkpoint():
    if not os.path.isdir(MODEL_DIR):
        return None
    for name in sorted(os.listdir(MODEL_DIR)):
        low = name.lower()
        if low.startswith("best_efn") and low.endswith((".pt", ".pth", ".pkl")):
            return os.path.join(MODEL_DIR, name)
    return None


def load_model():
    path = find_checkpoint()
    if path is None:
        raise FileNotFoundError(
            "No checkpoint found. Put your model in app/models/ named best_efn.pt or best_efn.pkl."
        )
    try:
        obj = torch.load(path, map_location=device, weights_only=True)  # safest option
    except Exception:
        # Full-module pickles cannot be loaded in safe mode. Only do this for YOUR OWN file.
        log.warning("Safe load failed, retrying with weights_only=False (trust this file only).")
        obj = torch.load(path, map_location=device, weights_only=False)

    if isinstance(obj, nn.Module):
        model = obj
    elif isinstance(obj, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            if isinstance(obj.get(key), dict):
                obj = obj[key]
                break
        state = {k.removeprefix("module."): v for k, v in obj.items()}
        model = build_model()
        model.load_state_dict(state)
    else:
        raise RuntimeError(f"Unrecognized checkpoint format in {path}")
    model.to(device).eval()
    return model, path


model, model_path = load_model()
log.info("Loaded model from %s on %s", model_path, device)
explainer = lime_image.LimeImageExplainer(random_state=42)


# --- Prediction helpers ---
def predict_proba(pil_images):
    """List of PIL RGB images -> array (N, 2) = [P(DR), P(No_DR)]."""
    batch = torch.stack([test_tf(img) for img in pil_images]).to(device)
    with torch.no_grad():
        p_no_dr = torch.sigmoid(model(batch).squeeze(1)).cpu().numpy()
    return np.stack([1.0 - p_no_dr, p_no_dr], axis=1)


def prune_outputs():
    files = [os.path.join(OUTPUT_DIR, f) for f in os.listdir(OUTPUT_DIR) if f.startswith("lime_")]
    files.sort(key=os.path.getmtime)
    for old in files[:-MAX_OUTPUT_FILES]:
        try:
            os.remove(old)
        except OSError:
            pass


def run_lime(pil_img, class_index):
    """Explain the predicted class. Returns the URL of the saved overlay image."""
    small = np.array(pil_img.resize((IMG_SIZE, IMG_SIZE)))  # fast: explain the size the model sees

    def batch_predict(arrs):
        return predict_proba([Image.fromarray(a.astype(np.uint8)) for a in arrs])

    exp = explainer.explain_instance(
        small, batch_predict, labels=(class_index,), num_samples=LIME_SAMPLES, hide_color=0
    )
    img, mask = exp.get_image_and_mask(
        label=class_index, positive_only=False, hide_rest=False, num_features=8
    )
    overlay = mark_boundaries(img, mask)
    name = f"lime_{uuid.uuid4().hex}.png"
    Image.fromarray((overlay * 255).astype(np.uint8)).save(os.path.join(OUTPUT_DIR, name))
    prune_outputs()
    return f"/static/outputs/{name}"


# --- Routes ---
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/health")
def health():
    return jsonify({"status": "ok", "model": os.path.basename(model_path), "device": str(device)})


@app.errorhandler(413)
def too_large(_):
    return jsonify({"error": "Image is too large. Upload a file under 10 MB."}), 413


@app.route("/predict", methods=["POST"])
def predict():
    file = request.files.get("image")
    if file is None or file.filename == "":
        return jsonify({"error": "Choose an image and upload it with the form field name 'image'."}), 400
    try:
        pil = Image.open(io.BytesIO(file.read())).convert("RGB")
    except Exception:
        return jsonify({"error": "This file is not a valid image. Use a JPG or PNG fundus photo."}), 400

    p_dr, p_no_dr = predict_proba([pil])[0]
    idx = 1 if p_no_dr >= THRESHOLD else 0
    result = {
        "prediction": CLASS_NAMES[idx],
        "probabilities": {"DR": round(float(p_dr), 4), "No_DR": round(float(p_no_dr), 4)},
        "disclaimer": "Research demo only. Not a medical diagnosis.",
    }
    try:
        result["lime_url"] = run_lime(pil, idx)
    except Exception:
        log.exception("LIME failed")
        result["lime_url"] = None  # still return the prediction
    return jsonify(result)


if __name__ == "__main__":
    # Local use only. Set FLASK_DEBUG=1 for debug mode. Use gunicorn/waitress for deployment.
    app.run(
        host=os.getenv("HOST", "127.0.0.1"),
        port=int(os.getenv("PORT", "5000")),
        debug=os.getenv("FLASK_DEBUG") == "1",
    )
