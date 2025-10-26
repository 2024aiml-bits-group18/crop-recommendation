# main.py – i18n + flow + predict (Top-K) + districts + commodity prices enrichment
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Any, List, Optional, Tuple
import os, json, sqlite3

import numpy as np
import pandas as pd

# Try both joblib and pickle gracefully
try:
    import joblib
except Exception:
    joblib = None
import pickle

API_BASE = "/api"
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "final_crop_model.pkl")   # model (either bundle or bare estimator)
I18N_DIR = os.path.join(BASE_DIR, "i18n")
FLOW_DIR = os.path.join(BASE_DIR, "flow")
FLOW_PATH = os.path.join(BASE_DIR, "flow.json")               # legacy support
TOP_K = 3

# ---------- SQLite: commodity min/max prices ----------
DB_PATH = os.path.join(BASE_DIR, "commodity_prices.sqlite3")
PRICE_TABLE = "commodity_prices"

# Predefined districts list (static fallback)
DISTRICTS_HARDCODED = [
    "Ananthapur", "Chittoor", "East Godavari", "Guntur", "Kadapa", "Krishna",
    "Kurnool", "Nellore", "Prakasam", "Srikakulam", "Visakhapatnam",
    "Vizianagaram", "West Godavari"
]

# ---------- Optional “artifact set” paths (if not using bundle) ----------
SCALER_PATH      = os.path.join(BASE_DIR, "scaler.pkl")
ENCODERS_PATH    = os.path.join(BASE_DIR, "encoders.pkl")      # dict: {col -> LabelEncoder}
COLUMNS_PATH     = os.path.join(BASE_DIR, "columns.json")      # list[str] exact X column order
LE_CROP_PATH     = os.path.join(BASE_DIR, "label_encoder.pkl") # LabelEncoder for target classes

# Columns (used by both modes to accept payload)
NUMERIC_FIELDS = [
    "pH","EC","OC","Avail-P","Exch-K","Avail-Ca","Avail-Mg","Avail-S",
    "Avail-Zn","Avail-B","Avail-Fe","Avail-Cu","Avail-Mn",
    "Kharif_rain","Rabi_rain","Zaid_rain"
]
CATEGORICAL_FIELDS = ["District","Soil_Type_Standard"]

MASTER_SOIL = [
    "Black","Red","Sandy","Loam","Clay","Brown","Yellow","White",
    "Laterite","Saline","Alkaline","Alluvial","Gravel/Stony","Mixed","Other","Unknown"
]

# -----------------------------------------------------------------------------
# FastAPI
# -----------------------------------------------------------------------------
app = FastAPI(title="Crop Recommender (Multilingual + Flow + Prices)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten for prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def _read_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _list_json_names(dirpath: str) -> List[str]:
    if not os.path.isdir(dirpath):
        return []
    return sorted([os.path.splitext(f)[0] for f in os.listdir(dirpath) if f.endswith(".json")])

def _flow_path_for_lang(lang: str) -> Optional[str]:
    for p in [
        os.path.join(FLOW_DIR, f"{lang}.json"),
        os.path.join(FLOW_DIR, f"flow_{lang}.json"),
        os.path.join(FLOW_DIR, "default.json"),
        os.path.join(FLOW_DIR, "flow.json"),
    ]:
        if os.path.exists(p):
            return p
    return None

def safe_float(x, default=0.0):
    try:
        if x is None or str(x).strip() == "":
            return float(default)
        return float(x)
    except Exception:
        return float(default)

def safe_mul(a, b) -> float:
    return safe_float(a, 0.0) * safe_float(b, 0.0)

def standardize_soil(raw: Any) -> str:
    if raw is None or str(raw).strip() == "":
        return "Unknown"
    txt = str(raw).strip().title()
    return txt if txt in MASTER_SOIL else txt

def topk_labels_from_probs(probs: np.ndarray, classes: List[str], k: int) -> List[str]:
    idx = np.argsort(probs)[::-1][:k]
    return [classes[i] for i in idx]

# -----------------------------------------------------------------------------
# PRICE CACHE
# -----------------------------------------------------------------------------
_prices_cache: Dict[str, Tuple[Optional[float], Optional[float]]] = {}

def _coerce_price(p: Optional[str]) -> Optional[float]:
    if p is None:
        return None
    s = str(p).strip().replace(",", "")
    try:
        return float(s)
    except Exception:
        return None

def load_prices_from_sqlite() -> Dict[str, Tuple[Optional[float], Optional[float]]]:
    if not os.path.exists(DB_PATH):
        print(f"⚠️ SQLite DB not found at {DB_PATH}. Price enrichment will return None.")
        return {}
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute(f"SELECT commodity, min_price, max_price FROM {PRICE_TABLE}")
        rows = cur.fetchall()
        conn.close()
    except Exception as e:
        print("⚠️ Failed to read prices from SQLite:", e)
        return {}

    out: Dict[str, Tuple[Optional[float], Optional[float]]] = {}
    for commodity, min_p, max_p in rows:
        out[str(commodity)] = (_coerce_price(min_p), _coerce_price(max_p))
    print(f"✅ Loaded {len(out)} commodity prices from SQLite")
    return out

_prices_cache = load_prices_from_sqlite()

def price_for(commodity: str) -> Tuple[Optional[float], Optional[float]]:
    return _prices_cache.get(commodity, (None, None))

# -----------------------------------------------------------------------------
# MODEL LOADING (supports both “bundle” and “artifact set”)
# -----------------------------------------------------------------------------
MODE = None  # "bundle" or "artifacts"

bundle = None
model = None
feature_names: Optional[List[str]] = None
scaler = None
encoders = None
le_crop = None
TRAIN_COLUMNS: Optional[List[str]] = None

def _exists(path: str) -> bool:
    return path and os.path.exists(path)

def _load_pickle(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)

# 1) Try legacy bundle
if _exists(MODEL_PATH) and joblib is not None:
    try:
        maybe = joblib.load(MODEL_PATH)
        if isinstance(maybe, dict) and "model" in maybe and "feature_names" in maybe:
            bundle = maybe
            model = bundle["model"]
            feature_names = list(bundle["feature_names"])
            MODE = "bundle"
            print("✅ Loaded legacy bundle model")
    except Exception as e:
        print("ℹ️ Bundle load failed/unsupported:", e)

# 2) Try artifact set
if MODE is None and _exists(MODEL_PATH):
    try:
        model = _load_pickle(MODEL_PATH)
        # require the rest:
        scaler = _load_pickle(SCALER_PATH) if _exists(SCALER_PATH) else None
        encoders = _load_pickle(ENCODERS_PATH) if _exists(ENCODERS_PATH) else None
        le_crop = _load_pickle(LE_CROP_PATH) if _exists(LE_CROP_PATH) else None
        if _exists(COLUMNS_PATH):
            with open(COLUMNS_PATH, "r") as f:
                TRAIN_COLUMNS = json.load(f)
        if scaler is not None and encoders is not None and le_crop is not None and TRAIN_COLUMNS:
            MODE = "artifacts"
            print("✅ Loaded artifact-set model + preprocessors")
        else:
            print("⚠️ Artifacts incomplete: need scaler.pkl, encoders.pkl, label_encoder.pkl, columns.json")
    except Exception as e:
        print("ℹ️ Artifact-set load failed:", e)

if MODE is None:
    print("❌ No usable model found. /api/predict will return 503.")

# -----------------------------------------------------------------------------
# BUILD ROW (bundle mode)
# -----------------------------------------------------------------------------
def build_row_bundle(payload: Dict[str, Any], feature_names_local: List[str]) -> pd.DataFrame:
    row: Dict[str, Any] = {}

    # numerics
    for f in NUMERIC_FIELDS:
        row[f] = safe_float(payload.get(f))

    # categoricals
    district = payload.get("District")
    row["District"] = str(district).strip() if district not in [None, ""] else "Unknown"
    sts_raw = payload.get("Soil_Type_Standard") or payload.get("Soil type")
    row["Soil_Type_Standard"] = standardize_soil(sts_raw)

    # example engineered interactions (must match what your bundle expects, else they’ll just be unused)
    interactions = {
        "pH_x_EC":       safe_mul(row.get("pH"), row.get("EC")),
        "OC_x_AvailP":   safe_mul(row.get("OC"), row.get("Avail-P")),
        "OC_x_ExchK":    safe_mul(row.get("OC"), row.get("Exch-K")),
        "pH_x_AvailCa":  safe_mul(row.get("pH"), row.get("Avail-Ca")),
        "EC_x_AvailFe":  safe_mul(row.get("EC"), row.get("Avail-Fe")),
    }

    data: Dict[str, List[Any]] = {}
    for col in feature_names_local:
        if col in row:
            data[col] = [row[col]]
        elif col in interactions:
            data[col] = [interactions[col]]
        elif col in CATEGORICAL_FIELDS:
            data[col] = ["Unknown"]
        else:
            data[col] = [0.0]

    df_one = pd.DataFrame(data, columns=feature_names_local)
    # basic coercions
    for col in feature_names_local:
        if col in CATEGORICAL_FIELDS:
            df_one[col] = df_one[col].astype("string").fillna("Unknown")
        else:
            df_one[col] = pd.to_numeric(df_one[col], errors="coerce").fillna(0.0).astype(float)
    return df_one

# -----------------------------------------------------------------------------
# ARTIFACT-SET PREPROCESS (encoders+scaler)
# -----------------------------------------------------------------------------
def safe_le_transform(le_obj, values: pd.Series) -> np.ndarray:
    known = set(le_obj.classes_.tolist())
    arr = values.fillna("").astype(str).apply(lambda x: x if x in known else None)
    fallback = le_obj.classes_[0] if len(le_obj.classes_) else ""
    arr = arr.fillna(fallback)
    return le_obj.transform(arr)

def align_and_transform_artifacts(payload: Dict[str, Any]) -> np.ndarray:
    # Build dataframe with incoming keys
    d: Dict[str, Any] = {k: payload.get(k) for k in (NUMERIC_FIELDS + CATEGORICAL_FIELDS)}
    # Map UI "Soil type" → Soil_Type_Standard if present
    if not d.get("Soil_Type_Standard") and payload.get("Soil type"):
        d["Soil_Type_Standard"] = payload.get("Soil type")
    d["Soil_Type_Standard"] = standardize_soil(d.get("Soil_Type_Standard"))
    d["District"] = (d.get("District") or "Unknown")

    df = pd.DataFrame([d])

    # Ensure all training columns exist
    for col in TRAIN_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    # Drop extras; exact order
    df = df[TRAIN_COLUMNS]

    # Encode categoricals
    for col in CATEGORICAL_FIELDS:
        if col in df.columns:
            le = encoders.get(col)
            if le is None:
                raise HTTPException(status_code=500, detail=f"Missing encoder for '{col}'.")
            df[col] = safe_le_transform(le, df[col])

    # Numerics → float with median fill
    for col in df.columns:
        if col not in CATEGORICAL_FIELDS:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            if df[col].isna().any():
                med = df[col].median()
                if pd.isna(med):
                    med = 0.0
                df[col] = df[col].fillna(med)

    # Scale
    X_scaled = scaler.transform(df.values)
    return X_scaled

# -----------------------------------------------------------------------------
# i18n (used by app.jsx)
# -----------------------------------------------------------------------------
@app.get(f"{API_BASE}/i18n/languages")
def get_languages():
    return {"languages": _list_json_names(I18N_DIR)}

@app.get(f"{API_BASE}/i18n/{{lang}}")
def get_language(lang: str):
    path = os.path.join(I18N_DIR, f"{lang}.json")
    if not os.path.exists(path):
        raise HTTPException(404, "Language not found")
    return _read_json(path)

# -----------------------------------------------------------------------------
# flow (used by app.jsx)
# -----------------------------------------------------------------------------
@app.get(f"{API_BASE}/flow", include_in_schema=False)
def get_flow_legacy(lang: Optional[str] = Query(default=None)):
    if lang:
        p = _flow_path_for_lang(lang)
        if not p:
            raise HTTPException(404, f"No flow file found for lang='{lang}' and no fallback present")
        return _read_json(p)
    if not os.path.exists(FLOW_PATH):
        return {"flow": []}
    return _read_json(FLOW_PATH)

@app.get(f"{API_BASE}/flow/languages")
def flow_languages():
    return {"languages": _list_json_names(FLOW_DIR)}

# -----------------------------------------------------------------------------
# meta: districts (static list)
# -----------------------------------------------------------------------------
@app.get(f"{API_BASE}/meta/districts")
def meta_districts():
    return {"districts": DISTRICTS_HARDCODED}

# -----------------------------------------------------------------------------
# health
# -----------------------------------------------------------------------------
@app.get(f"{API_BASE}/health")
def health():
    return {
        "status": "ok",
        "mode": MODE or "none",
        "model_loaded": (MODE is not None),
        "top_k": TOP_K,
        "i18n_languages": _list_json_names(I18N_DIR),
        "flow_languages": _list_json_names(FLOW_DIR),
        "prices_loaded": len(_prices_cache),
    }

# -----------------------------------------------------------------------------
# predict
# -----------------------------------------------------------------------------
@app.post(f"{API_BASE}/predict")
def predict(payload: Dict[str, Any]):
    if MODE is None:
        raise HTTPException(503, "Model not loaded")

    # Debug (optional)
    try:
        print("\n📥 Incoming /predict payload:\n", json.dumps(payload, indent=2))
    except Exception:
        pass

    # BUNDLE MODE
    if MODE == "bundle":
        X = build_row_bundle(payload, feature_names)
        probs = model.predict_proba(X)[0]
        classes = model.classes_.tolist()
        topk = topk_labels_from_probs(probs, classes, TOP_K)

    # ARTIFACT MODE
    else:
        X_scaled = align_and_transform_artifacts(payload)
        probs = model.predict_proba(X_scaled)[0]
        # classes are numeric indices; map back to crop names with le_crop
        idx = np.argsort(probs)[-TOP_K:][::-1]
        topk = le_crop.inverse_transform(idx).tolist()

    # Enrich with prices
    enriched = []
    for c in topk:
        min_p, max_p = price_for(c)
        enriched.append({"commodity": c, "min_price": min_p, "max_price": max_p})

    # Debug (optional)
    try:
        print("\n📤 Model top_k (enriched):\n", json.dumps(enriched, indent=2))
    except Exception:
        pass

    return {
        "top_k": topk,                 # legacy simple list
        "top_k_with_prices": enriched  # enriched objects list
    }
