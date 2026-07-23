"""
app.py
------
AI-Driven Crime Analytics & Visualization Platform
Main Streamlit entry point.

Run with:
    streamlit run app.py

This file wires together the other modules:
    - preprocessing.py  -> data cleaning & feature engineering
    - crime_model.py    -> Random Forest training / prediction / hotspot detection
    - visualization.py  -> Plotly charts & Folium heatmap
    - utils.py          -> theming, KPI helpers, column detection, export helpers
"""

from __future__ import annotations

import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

import crime_model as cm
import preprocessing as prep
import visualization as viz
from utils import (
    compute_kpis,
    dataframe_to_csv_bytes,
    format_number,
    inject_dark_theme,
    render_kpi_card,
)

# --------------------------------------------------------------------------- #
# PAGE CONFIG & THEME
# --------------------------------------------------------------------------- #

st.set_page_config(
    page_title="AI Crime Analytics Platform",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_dark_theme()


# --------------------------------------------------------------------------- #
# SESSION STATE INITIALIZATION
# --------------------------------------------------------------------------- #

def init_session_state() -> None:
    defaults = {
        "raw_df": None,
        "df": None,
        "columns_map": {},
        "training_result": None,
        "model_bundle": None,
        "prediction_df": None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


init_session_state()


# --------------------------------------------------------------------------- #
# SIDEBAR NAVIGATION
# --------------------------------------------------------------------------- #

st.sidebar.markdown(
    "<div class='app-title'>🚨 CrimeIQ</div>"
    "<p class='app-subtitle'>AI-Driven Crime Analytics Platform</p>",
    unsafe_allow_html=True,
)
st.sidebar.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)

PAGES = [
    "🏠 Dashboard",
    "📁 Upload & Preprocess Data",
    "📊 Visual Analytics",
    "🗺️ Hotspot Detection",
    "🤖 Train ML Model",
    "🔮 Predict Crime Category",
    "⬇️ Download Report",
]
page = st.sidebar.radio("Navigate", PAGES, label_visibility="collapsed")

st.sidebar.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
if st.session_state["df"] is not None:
    st.sidebar.success(f"Dataset loaded: {len(st.session_state['df']):,} records")
else:
    st.sidebar.info("No dataset loaded yet.")
if st.session_state["training_result"] is not None:
    st.sidebar.success(f"Model trained — accuracy {st.session_state['training_result'].accuracy:.2%}")

st.sidebar.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
st.sidebar.caption("Built with Streamlit • scikit-learn • Plotly • Folium")


# --------------------------------------------------------------------------- #
# PAGE: DASHBOARD (HOME)
# --------------------------------------------------------------------------- #

def page_dashboard():
    st.markdown(
        "<div class='app-title'>Crime Analytics Dashboard</div>"
        "<p class='app-subtitle'>Overview of key crime statistics from the loaded dataset</p>",
        unsafe_allow_html=True,
    )
    st.write("")

    df = st.session_state["df"]
    if df is None:
        st.warning("No dataset loaded yet. Go to **📁 Upload & Preprocess Data** to get started.")
        st.markdown("""
        ### What this platform does
        - 📁 Upload any crime-incident CSV dataset
        - 🧹 Automatically clean & preprocess the data
        - 🤖 Train a Random Forest model to predict crime category
        - 🗺️ Detect and visualize crime hotspots
        - 📊 Explore interactive Plotly charts & Folium heatmaps
        - ⬇️ Export prediction reports as CSV
        """)
        return

    cols_map = st.session_state["columns_map"]
    kpis = compute_kpis(df, cols_map.get("crime_type"), cols_map.get("district"))

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(render_kpi_card("Total Crimes", format_number(kpis["total_crimes"])), unsafe_allow_html=True)
    with c2:
        st.markdown(render_kpi_card("Crime Types", format_number(kpis["crime_types"])), unsafe_allow_html=True)
    with c3:
        st.markdown(render_kpi_card("Districts Covered", format_number(kpis["districts"])), unsafe_allow_html=True)
    with c4:
        st.markdown(render_kpi_card("Most Common Crime", kpis["top_crime"]), unsafe_allow_html=True)

    st.write("")
    c5, c6 = st.columns(2)
    with c5:
        st.markdown(render_kpi_card("Most Affected District", kpis["top_district"]), unsafe_allow_html=True)
    with c6:
        model_status = "Trained ✅" if st.session_state["training_result"] else "Not trained yet"
        st.markdown(render_kpi_card("ML Model Status", model_status), unsafe_allow_html=True)

    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
    st.subheader("Dataset Preview")
    st.dataframe(df.head(20), use_container_width=True)

    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
    col_a, col_b = st.columns(2)
    crime_col = cols_map.get("crime_type")
    district_col = cols_map.get("district")
    with col_a:
        if crime_col:
            st.plotly_chart(viz.bar_chart_crime_by_type(df, crime_col), use_container_width=True)
        else:
            st.info("No crime-type column detected for quick chart preview.")
    with col_b:
        if district_col:
            st.plotly_chart(viz.bar_chart_district(df, district_col), use_container_width=True)
        else:
            st.info("No district column detected for quick chart preview.")


# --------------------------------------------------------------------------- #
# PAGE: UPLOAD & PREPROCESS
# --------------------------------------------------------------------------- #

def page_upload():
    st.markdown(
        "<div class='app-title'>Upload & Preprocess Dataset</div>"
        "<p class='app-subtitle'>Upload a crime CSV file — cleaning & feature engineering runs automatically</p>",
        unsafe_allow_html=True,
    )
    st.write("")

    uploaded_file = st.file_uploader("Upload crime dataset (CSV)", type=["csv"])

    if uploaded_file is not None:
        try:
            raw_df = prep.load_data(uploaded_file)
            with st.spinner("Cleaning data and engineering features..."):
                processed_df, columns_map = prep.full_preprocessing_pipeline(raw_df)

            st.session_state["raw_df"] = raw_df
            st.session_state["df"] = processed_df
            st.session_state["columns_map"] = columns_map
            # Reset downstream state since a new dataset invalidates old model/predictions
            st.session_state["training_result"] = None
            st.session_state["model_bundle"] = None
            st.session_state["prediction_df"] = None

            st.success(f"Dataset loaded and preprocessed successfully: {len(processed_df):,} records.")
        except ValueError as err:
            st.error(str(err))
            return

    df = st.session_state["df"]
    if df is None:
        st.info("Upload a CSV file to begin. Typical columns: Date, Crime Type, District, Latitude, Longitude.")
        return

    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
    st.subheader("Detected Columns")
    cols_map = st.session_state["columns_map"]
    detected = {k: (v if v else "Not detected") for k, v in cols_map.items()}
    st.table(pd.DataFrame(detected.items(), columns=["Field", "Detected Column"]))

    st.subheader("Cleaned Data Preview")
    st.dataframe(df.head(30), use_container_width=True)

    st.subheader("Summary Statistics")
    st.dataframe(df.describe(include="all").transpose(), use_container_width=True)


# --------------------------------------------------------------------------- #
# PAGE: VISUAL ANALYTICS
# --------------------------------------------------------------------------- #

def page_visual_analytics():
    st.markdown(
        "<div class='app-title'>Visual Analytics</div>"
        "<p class='app-subtitle'>Interactive Plotly charts exploring crime patterns</p>",
        unsafe_allow_html=True,
    )
    st.write("")

    df = st.session_state["df"]
    if df is None:
        st.warning("Please upload a dataset first (📁 Upload & Preprocess Data).")
        return

    cols_map = st.session_state["columns_map"]
    crime_col = cols_map.get("crime_type")
    district_col = cols_map.get("district")

    tab1, tab2, tab3, tab4 = st.tabs(["Crime Types", "District Analysis", "Trends Over Time", "Time of Day"])

    with tab1:
        if crime_col:
            c1, c2 = st.columns(2)
            with c1:
                st.plotly_chart(viz.bar_chart_crime_by_type(df, crime_col), use_container_width=True)
            with c2:
                st.plotly_chart(viz.pie_chart_crime_distribution(df, crime_col), use_container_width=True)
        else:
            st.info("No crime-type column detected in this dataset.")

    with tab2:
        if district_col:
            st.plotly_chart(viz.bar_chart_district(df, district_col), use_container_width=True)
        else:
            st.info("No district/area column detected in this dataset.")

    with tab3:
        if "Crime_Year" in df.columns:
            st.plotly_chart(
                viz.line_chart_crime_trend(df, "Crime_Year", "Crime_Month"),
                use_container_width=True,
            )
        else:
            st.info("No date column detected — cannot plot trend over time.")

    with tab4:
        if "Crime_Hour" in df.columns and (df["Crime_Hour"] >= 0).any():
            st.plotly_chart(viz.bar_chart_hourly_distribution(df, "Crime_Hour"), use_container_width=True)
        else:
            st.info("No time-of-day information available in this dataset.")


# --------------------------------------------------------------------------- #
# PAGE: HOTSPOT DETECTION
# --------------------------------------------------------------------------- #

def page_hotspots():
    st.markdown(
        "<div class='app-title'>Crime Hotspot Detection</div>"
        "<p class='app-subtitle'>KMeans clustering + geographic heatmap of crime incidents</p>",
        unsafe_allow_html=True,
    )
    st.write("")

    df = st.session_state["df"]
    if df is None:
        st.warning("Please upload a dataset first (📁 Upload & Preprocess Data).")
        return

    cols_map = st.session_state["columns_map"]
    lat_col, lon_col = cols_map.get("latitude"), cols_map.get("longitude")

    if not lat_col or not lon_col:
        st.error(
            "Could not detect latitude/longitude columns in this dataset. "
            "Hotspot detection requires geographic coordinates."
        )
        return

    n_clusters = st.slider("Number of hotspot clusters (KMeans)", min_value=2, max_value=15, value=5)

    try:
        with st.spinner("Detecting hotspots..."):
            hotspot_df = cm.detect_hotspots(df, lat_col, lon_col, n_clusters=n_clusters)
    except ValueError as err:
        st.error(str(err))
        return

    st.subheader("Plotly Hotspot Map")
    color_col = cols_map.get("crime_type") if cols_map.get("crime_type") in hotspot_df.columns else "Hotspot_Cluster"
    st.plotly_chart(
        viz.scatter_map_hotspots(hotspot_df, lat_col, lon_col, color_col=color_col, hover_col=cols_map.get("crime_type")),
        use_container_width=True,
    )

    st.subheader("Folium Density Heatmap")
    try:
        fmap = viz.folium_crime_heatmap(df, lat_col, lon_col)
        st_folium(fmap, width=None, height=520, returned_objects=[])
    except ValueError as err:
        st.error(str(err))

    st.subheader("Top Hotspot Clusters by Incident Count")
    summary = (
        hotspot_df.groupby("Hotspot_Cluster")
        .agg(Incidents=("Hotspot_Cluster", "count"),
             Center_Lat=(lat_col, "mean"),
             Center_Lon=(lon_col, "mean"))
        .sort_values("Incidents", ascending=False)
        .reset_index()
    )
    st.dataframe(summary, use_container_width=True)


# --------------------------------------------------------------------------- #
# PAGE: TRAIN ML MODEL
# --------------------------------------------------------------------------- #

def page_train_model():
    st.markdown(
        "<div class='app-title'>Train Random Forest Model</div>"
        "<p class='app-subtitle'>Train a classifier to predict crime category from incident features</p>",
        unsafe_allow_html=True,
    )
    st.write("")

    df = st.session_state["df"]
    if df is None:
        st.warning("Please upload a dataset first (📁 Upload & Preprocess Data).")
        return

    cols_map = st.session_state["columns_map"]
    default_target = cols_map.get("crime_type")

    target_col = st.selectbox(
        "Select the target column to predict (crime category)",
        options=list(df.columns),
        index=list(df.columns).index(default_target) if default_target in df.columns else 0,
    )

    exclude_default = [c for c in [cols_map.get("description"), cols_map.get("date")] if c]
    exclude_cols = st.multiselect(
        "Columns to exclude from features (e.g. free-text or ID columns)",
        options=list(df.columns),
        default=exclude_default,
    )

    c1, c2 = st.columns(2)
    with c1:
        n_estimators = st.slider("Number of trees (n_estimators)", 50, 500, 200, step=50)
    with c2:
        test_size = st.slider("Test set size", 0.1, 0.4, 0.2, step=0.05)

    if st.button("🚀 Train Model", use_container_width=True):
        try:
            with st.spinner("Preparing features and training the Random Forest model..."):
                X, y, feature_names, encoders = prep.build_feature_target(
                    df, target_col=target_col, exclude_cols=exclude_cols
                )
                if X.empty or len(feature_names) == 0:
                    st.error("No usable feature columns remain after exclusions. Please adjust your selection.")
                    return
                result = cm.train_random_forest(
                    X, y, feature_names, encoders,
                    n_estimators=n_estimators, test_size=test_size,
                )
                saved_path = cm.save_model(result)

            st.session_state["training_result"] = result
            st.session_state["model_bundle"] = {
                "model": result.model,
                "encoders": result.encoders,
                "feature_names": result.feature_names,
                "class_labels": result.class_labels,
            }
            st.success(f"Model trained and saved to `{saved_path}` using joblib.")
        except Exception as err:  # noqa: BLE001
            st.error(f"Training failed: {err}")
            return

    result = st.session_state["training_result"]
    if result is not None:
        st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(render_kpi_card("Accuracy", f"{result.accuracy:.2%}"), unsafe_allow_html=True)
        with c2:
            st.markdown(render_kpi_card("Weighted F1-Score", f"{result.f1:.2%}"), unsafe_allow_html=True)

        st.write("")
        st.subheader("Classification Report")
        st.code(result.report)

        c3, c4 = st.columns(2)
        with c3:
            st.plotly_chart(
                viz.confusion_matrix_heatmap(result.confusion, result.class_labels),
                use_container_width=True,
            )
        with c4:
            st.plotly_chart(
                viz.bar_chart_feature_importance(result.feature_importances),
                use_container_width=True,
            )


# --------------------------------------------------------------------------- #
# PAGE: PREDICT CRIME CATEGORY
# --------------------------------------------------------------------------- #

def page_predict():
    st.markdown(
        "<div class='app-title'>Predict Crime Category</div>"
        "<p class='app-subtitle'>Use the trained Random Forest model to predict crime categories</p>",
        unsafe_allow_html=True,
    )
    st.write("")

    bundle = st.session_state.get("model_bundle")
    if bundle is None:
        if cm.model_exists():
            if st.button("Load previously saved model"):
                st.session_state["model_bundle"] = cm.load_model()
                st.rerun()
        st.warning("No trained model available yet. Go to **🤖 Train ML Model** first.")
        return

    df = st.session_state["df"]
    if df is None:
        st.warning("Please upload a dataset first (📁 Upload & Preprocess Data).")
        return

    st.info("Predictions will be generated for every row of the currently loaded dataset.")

    if st.button("🔮 Run Prediction on Loaded Dataset", use_container_width=True):
        with st.spinner("Generating predictions..."):
            feature_names = bundle["feature_names"]
            X_new = df.copy()
            preds = cm.predict_crime_category(bundle, X_new)
            result_df = df.copy()
            result_df["Predicted_Crime_Category"] = preds.values
            st.session_state["prediction_df"] = result_df
        st.success("Predictions generated successfully.")

    pred_df = st.session_state.get("prediction_df")
    if pred_df is not None:
        st.subheader("Prediction Results")
        st.dataframe(pred_df.head(50), use_container_width=True)

        st.subheader("Predicted Category Distribution")
        st.plotly_chart(
            viz.bar_chart_crime_by_type(pred_df, "Predicted_Crime_Category"),
            use_container_width=True,
        )


# --------------------------------------------------------------------------- #
# PAGE: DOWNLOAD REPORT
# --------------------------------------------------------------------------- #

def page_download():
    st.markdown(
        "<div class='app-title'>Download Prediction Report</div>"
        "<p class='app-subtitle'>Export your predictions or processed dataset as CSV</p>",
        unsafe_allow_html=True,
    )
    st.write("")

    pred_df = st.session_state.get("prediction_df")
    df = st.session_state.get("df")

    if pred_df is not None:
        st.success(f"Prediction report ready: {len(pred_df):,} rows.")
        st.download_button(
            "⬇️ Download Prediction Report (CSV)",
            data=dataframe_to_csv_bytes(pred_df),
            file_name="crime_prediction_report.csv",
            mime="text/csv",
            use_container_width=True,
        )
        st.dataframe(pred_df.head(20), use_container_width=True)
    else:
        st.warning("No predictions generated yet. Go to **🔮 Predict Crime Category** first.")

    if df is not None:
        st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
        st.subheader("Or export the cleaned/processed dataset")
        st.download_button(
            "⬇️ Download Cleaned Dataset (CSV)",
            data=dataframe_to_csv_bytes(df),
            file_name="crime_data_cleaned.csv",
            mime="text/csv",
            use_container_width=True,
        )


# --------------------------------------------------------------------------- #
# ROUTER
# --------------------------------------------------------------------------- #

ROUTES = {
    "🏠 Dashboard": page_dashboard,
    "📁 Upload & Preprocess Data": page_upload,
    "📊 Visual Analytics": page_visual_analytics,
    "🗺️ Hotspot Detection": page_hotspots,
    "🤖 Train ML Model": page_train_model,
    "🔮 Predict Crime Category": page_predict,
    "⬇️ Download Report": page_download,
}

ROUTES[page]()