# 🚨 CrimeIQ — AI-Driven Crime Analytics & Visualization Platform

An end-to-end, production-ready **Streamlit** web application for exploring crime
data, detecting hotspots, and predicting crime categories using a
**Random Forest** machine learning model — with a professional, responsive,
dark-themed dashboard UI.

---

## ✨ Features

| Feature | Description |
|---|---|
| 📁 **CSV Upload** | Upload any crime-incident CSV dataset |
| 🧹 **Automated Preprocessing** | Cleans missing values, duplicates, and engineers date/time features with Pandas |
| 🧠 **Smart Column Detection** | Auto-detects crime type, date, district, latitude/longitude columns from varied dataset schemas |
| 🤖 **Random Forest Classifier** | Trains a `scikit-learn` model to predict crime category from incident features |
| 💾 **Model Persistence** | Trained model + encoders saved/loaded with `joblib` |
| 🗺️ **Hotspot Detection** | KMeans clustering identifies geographic crime hotspots |
| 🌡️ **Folium Heatmap** | Interactive density heatmap of crime locations |
| 📊 **Interactive Plotly Charts** | Bar, pie, line, hourly-distribution, feature-importance, and confusion-matrix charts |
| 📈 **KPI Dashboard** | Total crimes, crime types, districts covered, top crime, top district |
| ⬇️ **Downloadable Reports** | Export predictions or cleaned data as CSV |
| 🎨 **Dark, Responsive UI** | Custom CSS + native Streamlit dark theme, sidebar navigation |

---

## 🗂️ Project Structure

```
crime-analytics-platform/
│
├── app.py                    # Main Streamlit app (UI, routing, session state)
├── preprocessing.py          # Data loading, cleaning, feature engineering
├── crime_model.py            # Random Forest training, prediction, hotspot detection, joblib I/O
├── visualization.py          # Plotly charts + Folium heatmap builders
├── utils.py                  # Theming, KPI calculations, column detection, export helpers
│
├── requirements.txt          # Python dependencies
├── README.md                 # This file
│
├── .streamlit/
│   └── config.toml           # Native Streamlit dark theme configuration
│
├── sample_data/
│   └── sample_crime_data.csv # Example dataset to try the app immediately
│
├── saved_models/             # Trained models are saved here via joblib (.joblib)
│   └── .gitkeep
│
└── .gitignore
```

---

## ⚙️ Setup Instructions (VS Code / Local)

### 1. Clone or download the project
```bash
git clone <your-repo-url>
cd crime-analytics-platform
```

### 2. Create and activate a virtual environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the app
```bash
streamlit run app.py
```

The app will open automatically at **http://localhost:8501**.

### 5. Try it out
- Use the built-in file uploader on the **📁 Upload & Preprocess Data** page.
- No dataset handy? Upload `sample_data/sample_crime_data.csv` to explore every feature immediately.

---

## 📄 Expected CSV Format

The app auto-detects columns by name, so exact headers aren't required — but
for best results include columns similar to:

| Column meaning | Example header names recognized |
|---|---|
| Crime category | `Crime Type`, `Category`, `Primary_Type`, `Offense` |
| Date/time | `Date`, `Occurred`, `Reported_Date`, `Datetime` |
| District/area | `District`, `Area`, `Zone`, `Precinct`, `Beat` |
| Latitude | `Latitude`, `Lat` |
| Longitude | `Longitude`, `Lon`, `Lng` |
| Description (optional) | `Description`, `Narrative` |

If a column can't be auto-detected (e.g. no coordinates in your dataset),
the app gracefully disables the features that depend on it (e.g. hotspot
detection) while the rest of the platform keeps working.

---

## 🧭 App Walkthrough (Sidebar Pages)

1. **🏠 Dashboard** — KPI summary cards, dataset preview, quick charts.
2. **📁 Upload & Preprocess Data** — Upload CSV, view detected columns, cleaned preview, and summary statistics.
3. **📊 Visual Analytics** — Tabs for crime-type breakdown, district analysis, trends over time, and time-of-day distribution (all Plotly, interactive).
4. **🗺️ Hotspot Detection** — KMeans-based clustering, Plotly scatter map, Folium density heatmap, and a ranked hotspot table.
5. **🤖 Train ML Model** — Choose the target column and features, train a Random Forest, and review accuracy, F1-score, confusion matrix, and feature importances. The model is saved via `joblib`.
6. **🔮 Predict Crime Category** — Run the trained model against the loaded dataset to generate predictions.
7. **⬇️ Download Report** — Export the prediction results or the cleaned dataset as CSV.

---

## 🧠 Machine Learning Details

- **Algorithm:** `RandomForestClassifier` (scikit-learn), with `class_weight="balanced"` to handle imbalanced crime categories.
- **Feature engineering:** Date columns are decomposed into `Crime_Year`, `Crime_Month`, `Crime_Day`, `Crime_Hour`, and `Crime_DayOfWeek`.
- **Encoding:** Categorical features are label-encoded; encoders are persisted alongside the model so new/unseen data can be transformed consistently at inference time.
- **Evaluation:** Accuracy, weighted F1-score, full classification report, and a confusion-matrix heatmap.
- **Interpretability:** Feature importance ranking is plotted after every training run.
- **Persistence:** The trained model, encoders, and feature list are bundled and saved to `saved_models/crime_rf_model.joblib` with `joblib.dump` / loaded with `joblib.load`.

## 🗺️ Hotspot Detection Details

- Uses `sklearn.cluster.KMeans` on `(latitude, longitude)` pairs.
- The number of clusters is adjustable via a slider (2–15).
- Clusters are ranked by incident count (`Hotspot_Rank`, 0 = most severe).
- Visualized both as an interactive Plotly scatter map and a Folium heatmap with clustered markers.

---

## 🚀 Deploying

This app is ready to push to GitHub and deploy on **Streamlit Community Cloud**,
or any platform that supports Streamlit apps (Render, Railway, Docker, etc.):

```bash
git init
git add .
git commit -m "Initial commit: CrimeIQ crime analytics platform"
git branch -M main
git remote add origin <your-repo-url>
git push -u origin main
```

Then connect the repo at [share.streamlit.io](https://share.streamlit.io) and
set the entry point to `app.py`.

---

## 🛠️ Tech Stack

- **Frontend/App Framework:** Streamlit
- **Data Processing:** Pandas, NumPy
- **Machine Learning:** scikit-learn (RandomForestClassifier, KMeans, LabelEncoder)
- **Visualization:** Plotly Express/Graph Objects, Folium + streamlit-folium
- **Model Persistence:** joblib

---

## 📌 Notes

- All prediction/report data stays local to your session — no external API calls are made.
- Retraining overwrites the saved model at `saved_models/crime_rf_model.joblib`.
- Large datasets: the Folium heatmap samples up to 500 points for individual markers (the heat layer itself uses all points) to keep the map responsive.

---

## 📃 License

This project is provided as-is for educational and analytical purposes. Adapt freely for your own crime-analytics use cases.