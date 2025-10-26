from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict
import json, os
import pandas as pd
import joblib
import random

API_BASE = "/api"
BASE_DIR = os.path.dirname(__file__)
I18N_DIR = os.path.join(BASE_DIR, "i18n")
FLOW_PATH = os.path.join(BASE_DIR, "flow.json")
MODEL_PATH = os.path.join(BASE_DIR, "knn_crop_model.joblib")  # put your model here

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# Try to load your real model (sklearn pipeline)
model = None
if os.path.exists(MODEL_PATH):
    try:
        model = joblib.load(MODEL_PATH)
        print("✅ Model loaded")
    except Exception as e:
        print("⚠️ Could not load model:", e)

@app.get(f"{API_BASE}/i18n/languages")
def get_languages():
    langs = [os.path.splitext(f)[0] for f in os.listdir(I18N_DIR) if f.endswith(".json")]
    return {"languages": sorted(langs)}

@app.get(f"{API_BASE}/i18n/{{lang}}")
def get_language(lang: str):
    path = os.path.join(I18N_DIR, f"{lang}.json")
    if not os.path.exists(path):
        raise HTTPException(404, "Language not found")
    return read_json(path)

@app.get(f"{API_BASE}/flow")
def get_flow():
    return read_json(FLOW_PATH)

@app.post(f"{API_BASE}/predict")
def predict(payload: Dict):
    # Build a row exactly like training columns (names must match your model)
    row = {
        "pH": payload.get("pH"),
        "EC": payload.get("EC"),
        "OC": payload.get("OC"),
        "Avail-P": payload.get("Avail-P"),
        "Exch-K": payload.get("Exch-K"),
        "Avail-Ca": payload.get("Avail-Ca"),
        "Avail-Mg": payload.get("Avail-Mg"),
        "Avail-S": payload.get("Avail-S"),
        "Avail-Zn": payload.get("Avail-Zn"),
        "Avail-B": payload.get("Avail-B"),
        "Avail-Fe": payload.get("Avail-Fe"),
        "Avail-Cu": payload.get("Avail-Cu"),
        "Avail-Mn": payload.get("Avail-Mn"),
        "Soil type": payload.get("Soil type"),
        "Season": payload.get("Season"),
        # optional extra context:
        "latitude": payload.get("latitude"),
        "longitude": payload.get("longitude"),
        "lang": payload.get("lang"),
    }
    df = pd.DataFrame([row])

    # If real model is loaded, use it
    if model is not None:
        pred = model.predict(df)[0]
        proba = {}
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(df)[0]
            classes = model.classes_
            proba = {str(c): float(p) for c, p in zip(classes, probs)}
        return {"prediction": str(pred), "probabilities": proba}

    # Demo fallback (so you can test UI without a trained model)
    demo_choices = ["Paddy", "Maize", "Cotton", "Groundnut"]
    choice = random.choice(demo_choices)
    probs = {c: (0.7 if c == choice else 0.1) for c in demo_choices}
    return {"prediction": choice, "probabilities": probs}
