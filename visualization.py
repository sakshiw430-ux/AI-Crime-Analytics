"""
visualization.py
-----------------
All charting and mapping logic for the Crime Analytics Platform.

Uses:
    - Plotly Express / Graph Objects for interactive bar, pie, line and
      scatter-map charts, styled to match the app's dark theme.
    - Folium (+ folium.plugins.HeatMap) for a geographic crime density
      heatmap, rendered inside Streamlit via streamlit-folium.
"""

from __future__ import annotations

from typing import Optional

import folium
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from folium.plugins import HeatMap, MarkerCluster

# Consistent dark-theme color palette used across all charts
DARK_TEMPLATE = "plotly_dark"
ACCENT_COLORS = ["#ff4b4b", "#ff9d00", "#ffd23f", "#3fa7d6", "#7b61ff", "#2ec4b6", "#ff6f91"]
PAPER_BG = "#0e1117"
PLOT_BG = "#12151c"


def _apply_dark_layout(fig: go.Figure, title: str) -> go.Figure:
    """Apply consistent dark-theme styling to any Plotly figure."""
    fig.update_layout(
        template=DARK_TEMPLATE,
        title=dict(text=title, x=0.02, font=dict(size=18, color="#f5f5f5")),
        paper_bgcolor=PAPER_BG,
        plot_bgcolor=PLOT_BG,
        font=dict(color="#d0d3d9"),
        margin=dict(l=30, r=20, t=60, b=30),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    return fig


# --------------------------------------------------------------------------- #
# BAR / PIE / LINE CHARTS
# --------------------------------------------------------------------------- #

def bar_chart_crime_by_type(df: pd.DataFrame, crime_col: str, top_n: int = 12) -> go.Figure:
    """Horizontal bar chart of the most frequent crime categories."""
    counts = df[crime_col].value_counts().head(top_n).reset_index()
    counts.columns = [crime_col, "Count"]
    fig = px.bar(
        counts.sort_values("Count"),
        x="Count", y=crime_col, orientation="h",
        color="Count", color_continuous_scale="OrRd",
        text="Count",
    )
    fig.update_traces(textposition="outside")
    return _apply_dark_layout(fig, f"Top {top_n} Crime Types")


def pie_chart_crime_distribution(df: pd.DataFrame, crime_col: str, top_n: int = 8) -> go.Figure:
    """Pie chart showing the proportional share of the leading crime types."""
    counts = df[crime_col].value_counts().head(top_n).reset_index()
    counts.columns = [crime_col, "Count"]
    fig = px.pie(
        counts, names=crime_col, values="Count",
        color_discrete_sequence=ACCENT_COLORS, hole=0.45,
    )
    fig.update_traces(textinfo="percent+label")
    return _apply_dark_layout(fig, "Crime Category Distribution")


def line_chart_crime_trend(df: pd.DataFrame, year_col: str, month_col: Optional[str] = None) -> go.Figure:
    """
    Line chart of crime volume over time. If a month column is present,
    trends are shown per Year-Month; otherwise aggregated by year only.
    """
    data = df[df[year_col] > 0].copy()
    if month_col and month_col in data.columns:
        data = data[data[month_col] > 0]
        data["Period"] = (
            data[year_col].astype(int).astype(str) + "-"
            + data[month_col].astype(int).astype(str).str.zfill(2)
        )
        trend = data.groupby("Period").size().reset_index(name="Count").sort_values("Period")
        x_col = "Period"
    else:
        trend = data.groupby(year_col).size().reset_index(name="Count").sort_values(year_col)
        x_col = year_col

    fig = px.line(trend, x=x_col, y="Count", markers=True)
    fig.update_traces(line=dict(color="#ff4b4b", width=3), marker=dict(size=6, color="#ff9d00"))
    return _apply_dark_layout(fig, "Crime Trend Over Time")


def bar_chart_district(df: pd.DataFrame, district_col: str, top_n: int = 12) -> go.Figure:
    """Bar chart of crime counts by district / area."""
    counts = df[district_col].value_counts().head(top_n).reset_index()
    counts.columns = [district_col, "Count"]
    fig = px.bar(
        counts, x=district_col, y="Count",
        color="Count", color_continuous_scale="Sunset",
        text="Count",
    )
    fig.update_traces(textposition="outside")
    fig.update_xaxes(tickangle=-40)
    return _apply_dark_layout(fig, f"Top {top_n} Districts by Crime Count")


def bar_chart_hourly_distribution(df: pd.DataFrame, hour_col: str) -> go.Figure:
    """Bar chart showing crime frequency by hour of day (0-23)."""
    data = df[df[hour_col] >= 0]
    counts = data.groupby(hour_col).size().reset_index(name="Count")
    fig = px.bar(
        counts, x=hour_col, y="Count",
        color="Count", color_continuous_scale="Plasma",
    )
    fig.update_layout(xaxis=dict(dtick=1))
    return _apply_dark_layout(fig, "Crime Frequency by Hour of Day")


# --------------------------------------------------------------------------- #
# GEOGRAPHIC VISUALIZATIONS
# --------------------------------------------------------------------------- #

def scatter_map_hotspots(
    df: pd.DataFrame, lat_col: str, lon_col: str,
    color_col: Optional[str] = None, hover_col: Optional[str] = None,
) -> go.Figure:
    """
    Interactive Plotly scatter-map of crime incidents, dark-themed.

    Uses the newer `px.scatter_map` (MapLibre-based, Plotly >= 5.24) when
    available, and transparently falls back to the legacy `px.scatter_mapbox`
    on older Plotly versions.
    """
    data = df.dropna(subset=[lat_col, lon_col])
    color_arg = color_col if (color_col and color_col in data.columns) else None
    hover_arg = hover_col if (hover_col and hover_col in data.columns) else None
    scale_arg = "OrRd" if color_arg else None

    use_new_api = hasattr(px, "scatter_map")
    map_fn = px.scatter_map if use_new_api else px.scatter_mapbox

    fig = map_fn(
        data, lat=lat_col, lon=lon_col,
        color=color_arg, hover_name=hover_arg,
        zoom=9, height=550,
        color_continuous_scale=scale_arg,
    )

    if use_new_api:
        fig.update_layout(map_style="dark")
    else:
        fig.update_layout(mapbox_style="carto-darkmatter")

    fig = _apply_dark_layout(fig, "Crime Hotspot Map")
    fig.update_layout(margin=dict(l=0, r=0, t=50, b=0))
    return fig


def folium_crime_heatmap(
    df: pd.DataFrame, lat_col: str, lon_col: str, zoom_start: int = 11,
) -> folium.Map:
    """
    Build a Folium map with a HeatMap layer showing crime density, plus a
    clustered marker layer for drill-down inspection of individual points.
    """
    data = df.dropna(subset=[lat_col, lon_col]).copy()
    data[lat_col] = pd.to_numeric(data[lat_col], errors="coerce")
    data[lon_col] = pd.to_numeric(data[lon_col], errors="coerce")
    data = data.dropna(subset=[lat_col, lon_col])

    if data.empty:
        raise ValueError("No valid coordinates available to build the heatmap.")

    center = [data[lat_col].mean(), data[lon_col].mean()]
    fmap = folium.Map(location=center, zoom_start=zoom_start, tiles="CartoDB dark_matter")

    heat_points = data[[lat_col, lon_col]].values.tolist()
    HeatMap(heat_points, radius=12, blur=18, min_opacity=0.4).add_to(fmap)

    # Cap marker cluster size for performance on large datasets
    cluster = MarkerCluster().add_to(fmap)
    sample = data.sample(min(len(data), 500), random_state=42)
    for _, row in sample.iterrows():
        folium.CircleMarker(
            location=[row[lat_col], row[lon_col]],
            radius=3, color="#ff9d00", fill=True, fill_opacity=0.7,
        ).add_to(cluster)

    return fmap


def bar_chart_feature_importance(importance_df: pd.DataFrame) -> go.Figure:
    """Bar chart of Random Forest feature importances (for model interpretability)."""
    top = importance_df.head(15).sort_values("importance")
    fig = px.bar(
        top, x="importance", y="feature", orientation="h",
        color="importance", color_continuous_scale="Turbo",
    )
    return _apply_dark_layout(fig, "Feature Importance (Random Forest)")


def confusion_matrix_heatmap(conf_matrix, class_labels) -> go.Figure:
    """Heatmap visualization of the model's confusion matrix."""
    fig = go.Figure(
        data=go.Heatmap(
            z=conf_matrix,
            x=class_labels,
            y=class_labels,
            colorscale="OrRd",
            showscale=True,
        )
    )
    fig.update_xaxes(title="Predicted")
    fig.update_yaxes(title="Actual")
    return _apply_dark_layout(fig, "Confusion Matrix")