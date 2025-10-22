# main.py – i18n + flow + predict (Top-K) + districts + commodity prices enrichment
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Any, List, Optional, Tuple
import os, json, joblib, pandas as pd, numpy as np, sqlite3

API_BASE = "/api"
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "serving_catboost_topN.pkl")  # your v4 model
I18N_DIR = os.path.join(BASE_DIR, "i18n")
FLOW_DIR = os.path.join(BASE_DIR, "flow")
FLOW_PATH = os.path.join(BASE_DIR, "flow.json")                   # legacy support
TOP_K = 3

# ---------- SQLite: commodity min/max prices ----------
DB_PATH = os.path.join(BASE_DIR, "commodity_prices.sqlite3")
PRICE_TABLE = "commodity_prices"

# Predefined districts list (from your earlier dataset; trimmed here for brevity)
DISTRICTS_HARDCODED = [
    "Ananthapur", "Chittoor", "East Godavari", "Guntur", "Kadapa", "Krishna",
    "Kurnool", "Nellore", "Prakasam", "Srikakulam", "Visakhapatnam",
    "Vizianagaram", "West Godavari"
]

app = FastAPI(title="Crop Recommender (Multilingual + Flow + Prices)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten for prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------ Load model bundle ------------------
bundle = None
if os.path.exists(MODEL_PATH):
    try:
        bundle = joblib.load(MODEL_PATH)
        print("✅ Model bundle loaded")
    except Exception as e:
        print("⚠️ Could not load model bundle:", e)

# ------------------ Utilities ------------------
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

MASTER_SOIL = [
    "Black","Red","Sandy","Loam","Clay","Brown","Yellow","White",
    "Laterite","Saline","Alkaline","Alluvial","Gravel/Stony","Mixed","Other","Unknown"
]

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

def build_row(payload: Dict[str, Any], feature_names: List[str]) -> pd.DataFrame:
    numeric_fields = [
        "pH","EC","OC","Avail-P","Exch-K","Avail-Ca","Avail-Mg","Avail-S",
        "Avail-Zn","Avail-B","Avail-Fe","Avail-Cu","Avail-Mn",
        "Kharif_rain","Rabi_rain","Zaid_rain"
    ]
    cat_fields = ["District","Soil_Type_Standard"]

    row: Dict[str, Any] = {}
    for f in numeric_fields:
        row[f] = safe_float(payload.get(f))

    district = payload.get("District")
    row["District"] = str(district).strip() if district not in [None, ""] else "Unknown"
    sts_raw = payload.get("Soil_Type_Standard") or payload.get("Soil type")
    row["Soil_Type_Standard"] = standardize_soil(sts_raw)

    interactions = {
        "pH_x_EC":       safe_mul(row.get("pH"), row.get("EC")),
        "OC_x_AvailP":   safe_mul(row.get("OC"), row.get("Avail-P")),
        "OC_x_ExchK":    safe_mul(row.get("OC"), row.get("Exch-K")),
        "pH_x_AvailCa":  safe_mul(row.get("pH"), row.get("Avail-Ca")),
        "EC_x_AvailFe":  safe_mul(row.get("EC"), row.get("Avail-Fe")),
    }

    data: Dict[str, List[Any]] = {}
    for col in feature_names:
        if col in row:
            data[col] = [row[col]]
        elif col in interactions:
            data[col] = [interactions[col]]
        else:
            data[col] = ["Unknown"] if col in cat_fields else [0.0]

    df_one = pd.DataFrame(data, columns=feature_names)
    for col in feature_names:
        if col in cat_fields:
            df_one[col] = df_one[col].astype("string").fillna("Unknown")
        else:
            df_one[col] = pd.to_numeric(df_one[col], errors="coerce").fillna(0.0).astype(float)
    return df_one

def topk_labels(probs: np.ndarray, classes: List[str], k: int) -> List[str]:
    idx = np.argsort(probs)[::-1][:k]
    return [classes[i] for i in idx]

# ---------- Prices cache (loaded from SQLite on startup) ----------
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
    """Reads all rows from SQLite into a dict: {commodity: (min_price, max_price)}"""
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
    """Lookup min/max from cache; fall back to None, None."""
    return _prices_cache.get(commodity, (None, None))

# ------------------ i18n (used by app.jsx) ------------------
@app.get(f"{API_BASE}/i18n/languages")
def get_languages():
    return {"languages": _list_json_names(I18N_DIR)}

@app.get(f"{API_BASE}/i18n/{{lang}}")
def get_language(lang: str):
    path = os.path.join(I18N_DIR, f"{lang}.json")
    if not os.path.exists(path):
        raise HTTPException(404, "Language not found")
    return _read_json(path)

# ------------------ flow (used by app.jsx) ------------------
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

# ------------------ meta: districts (static list) ------------------
@app.get(f"{API_BASE}/meta/districts")
def meta_districts():
    return {"districts": DISTRICTS_HARDCODED}

# ------------------ health ------------------
@app.get(f"{API_BASE}/health")
def health():
    return {
        "status": "ok",
        "model_loaded": bundle is not None,
        "top_k": TOP_K,
        "i18n_languages": _list_json_names(I18N_DIR),
        "flow_languages": _list_json_names(FLOW_DIR),
        "prices_loaded": len(_prices_cache),
    }

# ------------------ predict ------------------
@app.post(f"{API_BASE}/predict")
def predict(payload: Dict[str, Any]):
    if bundle is None:
        raise HTTPException(503, "Model bundle not loaded")

    # Debug (optional)
    try:
        print("\n📥 Incoming /predict payload:\n", json.dumps(payload, indent=2))
    except Exception:
        pass

    cb = bundle["model"]
    feature_names = bundle["feature_names"]

    X = build_row(payload, feature_names)
    probs = cb.predict_proba(X)[0]
    classes = cb.classes_.tolist()
    topk = topk_labels(probs, classes, TOP_K)

    # Enrich with min/max prices from SQLite (if available)
    enriched = []
    for c in topk:
        min_p, max_p = price_for(c)
        enriched.append({
            "commodity": c,
            "min_price": min_p,
            "max_price": max_p
        })

    # Debug (optional)
    try:
        print("\n📤 Model top_k (enriched):\n", json.dumps(enriched, indent=2))
    except Exception:
        pass

    # Backward-compatible + enriched payload
    return {
        "top_k": topk,                       # original list of strings (for older UI)
        "top_k_with_prices": enriched        # new enriched list of objects
    }
