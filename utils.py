"""
utils.py
--------
General-purpose helper utilities shared across the Crime Analytics Platform.

Contains:
    - Dark theme CSS injection for Streamlit
    - Smart column-name detection (crime type, date, district, lat/lon, etc.)
    - KPI calculation helpers
    - CSV export / download helpers
    - Small formatting helpers

Keeping these in one place avoids duplicating logic between app.py and the
other modules, and makes the app resilient to CSV files that use slightly
different column naming conventions.
"""

from __future__ import annotations

import io
from typing import Optional

import pandas as pd
import streamlit as st


# --------------------------------------------------------------------------- #
# THEME / STYLING
# --------------------------------------------------------------------------- #

def inject_dark_theme() -> None:
    """
    Inject custom CSS into the Streamlit app to create a professional,
    modern dark-themed dashboard look (cards, KPIs, sidebar, tables, etc.).
    Call this once, right after `st.set_page_config(...)` in app.py.
    """
    st.markdown(
        """
        <style>
        /* ---------- Global ---------- */
        .stApp {
            background-color: #0e1117;
            color: #e6e6e6;
        }
        html, body, [class*="css"] {
            font-family: 'Segoe UI', 'Inter', sans-serif;
        }

        /* ---------- Sidebar ---------- */
        section[data-testid="stSidebar"] {
            background-color: #12151c;
            border-right: 1px solid #262b36;
        }
        section[data-testid="stSidebar"] .stRadio label {
            font-size: 15px;
        }

        /* ---------- Headings ---------- */
        h1, h2, h3 {
            color: #f5f5f5;
            font-weight: 700;
        }
        .app-title {
            font-size: 2.1rem;
            font-weight: 800;
            background: linear-gradient(90deg, #ff4b4b, #ff9d00);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0;
        }
        .app-subtitle {
            color: #9aa0ab;
            font-size: 0.95rem;
            margin-top: 0;
        }

        /* ---------- KPI Cards ---------- */
        .kpi-card {
            background: linear-gradient(145deg, #161a23, #1c212c);
            border: 1px solid #262b36;
            border-radius: 14px;
            padding: 18px 20px;
            text-align: left;
            box-shadow: 0 4px 14px rgba(0,0,0,0.35);
        }
        .kpi-label {
            color: #9aa0ab;
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            margin-bottom: 6px;
        }
        .kpi-value {
            color: #ffffff;
            font-size: 1.9rem;
            font-weight: 800;
        }
        .kpi-accent {
            height: 4px;
            width: 42px;
            border-radius: 4px;
            background: linear-gradient(90deg, #ff4b4b, #ff9d00);
            margin-top: 10px;
        }

        /* ---------- Buttons ---------- */
        .stButton>button, .stDownloadButton>button {
            background: linear-gradient(90deg, #ff4b4b, #ff9d00);
            color: white;
            border: none;
            border-radius: 8px;
            padding: 0.5rem 1.2rem;
            font-weight: 600;
        }
        .stButton>button:hover, .stDownloadButton>button:hover {
            opacity: 0.88;
            color: white;
        }

        /* ---------- Dataframe ---------- */
        .stDataFrame {
            border-radius: 10px;
            overflow: hidden;
        }

        /* ---------- Section divider ---------- */
        .section-divider {
            border-top: 1px solid #262b36;
            margin: 1.2rem 0 1.4rem 0;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_kpi_card(label: str, value: str) -> str:
    """Return HTML markup for a single KPI card (used with st.markdown)."""
    return f"""
        <div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-accent"></div>
        </div>
    """


# --------------------------------------------------------------------------- #
# COLUMN DETECTION
# --------------------------------------------------------------------------- #

# Candidate name fragments (lower-case) used to auto-detect key columns in
# an arbitrary crime dataset. This lets the app work with many real-world
# datasets (Chicago, LA, India NCRB style exports, etc.) without the user
# having to rename columns manually.
_COLUMN_HINTS = {
    "crime_type": ["crime_type", "crimetype", "category", "primary_type", "offense", "crime_category", "type"],
    "date": ["date", "occurred", "reported_date", "datetime", "timestamp"],
    "district": ["district", "area", "zone", "precinct", "region", "location_desc", "beat", "city"],
    "latitude": ["lat", "latitude"],
    "longitude": ["lon", "lng", "long", "longitude"],
    "description": ["description", "desc", "narrative"],
}


def detect_column(df: pd.DataFrame, field: str) -> Optional[str]:
    """
    Best-effort detection of a semantic column (e.g. 'crime_type', 'latitude')
    within an arbitrary uploaded DataFrame, based on fuzzy name matching.

    Returns the matching column name, or None if nothing plausible was found.
    """
    hints = _COLUMN_HINTS.get(field, [])
    cols_lower = {c: c.lower().replace(" ", "_") for c in df.columns}

    # exact match first
    for col, lc in cols_lower.items():
        if lc in hints:
            return col
    # substring match next
    for col, lc in cols_lower.items():
        for hint in hints:
            if hint in lc:
                return col
    return None


def detect_all_columns(df: pd.DataFrame) -> dict:
    """Run detect_column for every known semantic field, return a dict map."""
    return {field: detect_column(df, field) for field in _COLUMN_HINTS}


# --------------------------------------------------------------------------- #
# KPI CALCULATIONS
# --------------------------------------------------------------------------- #

def compute_kpis(df: pd.DataFrame, crime_col: Optional[str], district_col: Optional[str]) -> dict:
    """
    Compute headline KPIs for the dashboard:
        - total number of crime records
        - number of distinct crime types
        - number of distinct districts / areas
        - most frequent crime type
        - most affected district
    """
    kpis = {
        "total_crimes": len(df),
        "crime_types": df[crime_col].nunique() if crime_col and crime_col in df.columns else 0,
        "districts": df[district_col].nunique() if district_col and district_col in df.columns else 0,
        "top_crime": "N/A",
        "top_district": "N/A",
    }
    if crime_col and crime_col in df.columns and not df[crime_col].empty:
        kpis["top_crime"] = str(df[crime_col].mode().iloc[0])
    if district_col and district_col in df.columns and not df[district_col].empty:
        kpis["top_district"] = str(df[district_col].mode().iloc[0])
    return kpis


# --------------------------------------------------------------------------- #
# EXPORT HELPERS
# --------------------------------------------------------------------------- #

def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    """Convert a DataFrame into UTF-8 encoded CSV bytes for st.download_button."""
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    return buffer.getvalue().encode("utf-8")


def format_number(n: int) -> str:
    """Format an integer with thousands separators, e.g. 12345 -> '12,345'."""
    try:
        return f"{int(n):,}"
    except (ValueError, TypeError):
        return str(n)