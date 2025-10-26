#!/usr/bin/env python3
"""
Fetch first-available min/max prices (no Arrival_Date filter) for Andhra Pradesh
commodities from data.gov.in (resource: 35985678-0d79-46b4-9ed6-6f13308a1d24),
then store ONLY (commodity, min_price, max_price) into a portable SQLite database.

Usage:
  - Fetch and store (overwriting existing rows):
      python prices_to_db.py "Rice,Maize,Onion"
  - Show what’s in DB:
      python prices_to_db.py --show

Environment:
  - DATA_GOV_IN_API_KEY   (optional; defaults to demo key in code)
"""

import os
import sys
import json
import time
import sqlite3
import requests
from typing import List, Dict, Optional

# -------- API CONFIG --------
API_KEY = os.getenv("DATA_GOV_IN_API_KEY", "579b464db66ec23bdd0000015458feb2622a43cd67760b22cb42145f")
RESOURCE_ID = "35985678-0d79-46b4-9ed6-6f13308a1d24"
BASE_URL = f"https://api.data.gov.in/resource/{RESOURCE_ID}"
STATE = "Andhra Pradesh"

# -------- DB CONFIG --------
DB_PATH = "commodity_prices.sqlite3"
TABLE_NAME = "commodity_prices"

# -------- DEFAULT INPUTS ----
DEFAULT_COMMODITIES = [
    "Pomegranate",
    "Rice",
    "Millets",
    "Banana",
    "Beans",
    "Ridgeguard(Tori)",
    "Bengal Gram(Gram)(Whole)",
    "Black Gram (Urd Beans)(Whole)",
    "Brinjal",
    "Cabbage",
    "Cashewnuts",
    "Castor Seed",
    "Coconut",
    "Corriander seed",
    "Cotton",
    "Cowpea (Lobia/Karamani)",
    "Cucumbar(Kheera)",
    "Season Leaves",
    "Little gourd (Kundru)",
    "Wood",
    "Tube Flower",
    "Green Fodder",
    "Green Gram (Moong)(Whole)",
    "Groundnut",
    "Kulthi(Horse Gram)",
    "Jowar(Sorghum)",
    "Maize",
    "Mango",
    "Ambady/Mesta",
    "Leafy Vegetable",
    "Onion",
    "Paddy(Dhan)(Common)",
    "Papaya",
    "Arhar (Tur/Red Gram)(Whole)",
    "Potato",
    "Fish",
    "Other Pulses",
    "Ragi (Finger Millet)",
    "Sesamum(Sesame,Gingelly,Til)",
    "Sugarcane",
    "Sunflower",
    "Soyabean",
    "Tobacco",
    "Tomato",
    "Tapioca",
    "Turmeric",
    "Other green and fresh vegetables",
    "Yam",
    "Bajra(Pearl Millet/Cumbu)",
    "Foxtail Millet(Navane)"
]


def get_first_record_for_commodity(commodity: str) -> Dict[str, Optional[str]]:
    """
    Call the API for a commodity without Arrival_Date filter.
    Return dict with ONLY commodity, min_price, max_price (strings from API).
    """
    params = {
        "api-key": API_KEY,
        "format": "json",
        "filters[State]": STATE,
        "filters[Commodity]": commodity,
        "limit": "1",
        "offset": "0",
    }
    try:
        resp = requests.get(BASE_URL, params=params, timeout=20)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        return {"commodity": commodity, "min_price": None, "max_price": None, "error": f"API error: {e}"}

    records = data.get("records") if isinstance(data, dict) else None
    if isinstance(records, list) and len(records) > 0:
        first = records[0]
        return {
            "commodity": commodity,
            "min_price": first.get("Min_Price"),
            "max_price": first.get("Max_Price"),
        }

    return {"commodity": commodity, "min_price": None, "max_price": None, "error": "No records returned."}

# ---------------- SQLite helpers ----------------

def ensure_table(conn: sqlite3.Connection) -> None:
    """
    Create table if not exists. Only the three requested columns.
    """
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
            commodity   TEXT PRIMARY KEY,
            min_price   TEXT,
            max_price   TEXT
        );
    """)
    conn.commit()

def clear_table(conn: sqlite3.Connection) -> None:
    """Delete all existing rows."""
    conn.execute(f"DELETE FROM {TABLE_NAME};")
    conn.commit()

def save_prices(conn: sqlite3.Connection, rows: List[Dict[str, Optional[str]]]) -> None:
    """
    Save list of dicts with keys: commodity, min_price, max_price.
    Assumes table exists and has been cleared.
    """
    # Use executemany for performance
    conn.executemany(
        f"INSERT OR REPLACE INTO {TABLE_NAME} (commodity, min_price, max_price) VALUES (?, ?, ?);",
        [(r.get("commodity"), r.get("min_price"), r.get("max_price")) for r in rows]
    )
    conn.commit()

def fetch_prices(conn: sqlite3.Connection) -> List[Dict[str, Optional[str]]]:
    """Return all rows as a list of dicts."""
    cur = conn.execute(f"SELECT commodity, min_price, max_price FROM {TABLE_NAME} ORDER BY commodity;")
    out = []
    for commodity, min_price, max_price in cur.fetchall():
        out.append({
            "commodity": commodity,
            "min_price": min_price,
            "max_price": max_price
        })
    return out

# --------------- Orchestration -------------------

def fetch_and_store(commodities: List[str]) -> List[Dict[str, Optional[str]]]:
    """
    Fetch first-available min/max for each commodity and store to SQLite,
    replacing existing rows each run.
    """
    results = []
    for c in commodities:
        results.append(get_first_record_for_commodity(c))
        time.sleep(0.25)  # gentle on API

    # Open DB, ensure table, clear, then save
    conn = sqlite3.connect(DB_PATH)
    try:
        ensure_table(conn)
        clear_table(conn)
        # Only store required keys
        to_store = [{"commodity": r["commodity"], "min_price": r.get("min_price"), "max_price": r.get("max_price")} for r in results]
        save_prices(conn, to_store)
    finally:
        conn.close()

    return results

def main():
    # CLI: --show to print DB; else fetch and store
    if len(sys.argv) > 1 and sys.argv[1].strip().lower() == "--show":
        conn = sqlite3.connect(DB_PATH)
        try:
            ensure_table(conn)
            rows = fetch_prices(conn)
        finally:
            conn.close()
        print(json.dumps(rows, indent=2))
        return

    # Parse commodity list from first arg, else use defaults
    if len(sys.argv) > 1 and sys.argv[1].strip() and sys.argv[1].strip() != "--show":
        commodities = [c.strip() for c in sys.argv[1].split(",") if c.strip()]
    else:
        commodities = DEFAULT_COMMODITIES

    results = fetch_and_store(commodities)

    # For visibility, also print what was fetched this run
    print(json.dumps(results, indent=2))

# ------------- Reusable function for other programs -------------

def get_prices_from_db(db_path: str = DB_PATH) -> List[Dict[str, Optional[str]]]:
    """
    Import this function in other programs to read the stored prices:
        from prices_to_db import get_prices_from_db
        rows = get_prices_from_db()
    Returns a list of dicts: {commodity, min_price, max_price}
    """
    conn = sqlite3.connect(db_path)
    try:
        ensure_table(conn)
        return fetch_prices(conn)
    finally:
        conn.close()

if __name__ == "__main__":
    main()
