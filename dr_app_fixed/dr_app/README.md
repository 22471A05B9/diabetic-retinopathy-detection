# Diabetic Retinopathy Detector (Flask + PyTorch + LIME)

A web app that takes a retinal fundus image, predicts **DR** or **No_DR** with an EfficientNet-B0 model, and shows a **LIME** overlay of the regions that influenced the prediction.

> **Research demo only.** This is not a medical device and does not give a diagnosis.

## Project structure
```
dr_app/
├─ app/
│  ├─ app.py            # Flask app: page, /predict, /health
│  ├─ models/best_efn.pkl   # trained checkpoint
│  ├─ static/           # app.js, style.css, outputs/ (generated LIME images)
│  └─ templates/index.html
├─ requirements.txt
└─ README.md
```

## Run locally
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate      macOS/Linux: source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
cd app
python app.py
```
Open http://127.0.0.1:5000

If `pip install torch` fails, copy the install command for your system from https://pytorch.org.

## Model checkpoint
Put your checkpoint in `app/models/` named `best_efn.pt`, `best_efn.pth` or `best_efn.pkl`. Accepted formats: a raw `state_dict`, a dict with `state_dict` / `model_state_dict`, or a full `nn.Module`. The model is EfficientNet-B0 with a single-logit head; the sigmoid output is read as P(No_DR). Check that this matches how you trained it.

## API
```bash
curl -X POST http://127.0.0.1:5000/predict -F "image=@fundus.jpg"
```
Returns `prediction`, `probabilities`, `lime_url` and a `disclaimer`. `GET /health` returns the model name and device.

## Settings
- `LIME_SAMPLES` (default 500): more samples give a steadier explanation but are slower.
- `FLASK_DEBUG=1` turns on debug mode (local use only).
- `HOST` and `PORT` change the address.

## Deploy (optional)
```bash
cd app
gunicorn -w 2 -b 0.0.0.0:5000 app:app        # Linux/macOS
waitress-serve --port=5000 app:app            # Windows
```

## Limits
- Uploads are limited to 10 MB.
- LIME runs on a 224x224 version of the image and is slow on a CPU.
- Results depend on the training data and are not clinically validated.
