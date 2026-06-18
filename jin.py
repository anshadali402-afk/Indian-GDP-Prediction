import json
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

# ────────────────────────────────────────────────────────────
# Page Config
# ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="India GDP Growth Predictor",
    page_icon="🇮🇳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ────────────────────────────────────────────────────────────
# Theming (dark, matches notebook palette)
# ────────────────────────────────────────────────────────────
PALETTE = ['#7c3aed', '#06b6d4', '#10b981', '#f59e0b', '#ef4444', '#ec4899']

st.markdown("""
<style>
.stApp { background-color: #0f0f1a; color: #e0e0ff; }
section[data-testid="stSidebar"] { background-color: #14142a; }
h1, h2, h3, h4 { color: #e0e0ff !important; }
div[data-testid="stMetric"] {
    background-color: #1a1a2e;
    border: 1px solid #2a2a4a;
    border-radius: 12px;
    padding: 14px;
}
div[data-testid="stMetricValue"] { color: #a78bfa; }
.stButton>button {
    background: linear-gradient(90deg, #7c3aed, #06b6d4);
    color: white; border: none; border-radius: 8px; font-weight: 600;
    padding: 0.6em 1.5em;
}
.stButton>button:hover { opacity: 0.85; }
.block-container { padding-top: 2rem; }
hr { border-color: #2a2a4a; }
</style>
""", unsafe_allow_html=True)


# ────────────────────────────────────────────────────────────
# Load artifacts
# ────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model = joblib.load("best_model.pkl")
    scaler = joblib.load("scaler.pkl")
    with open("feature_names.json") as f:
        feature_names = json.load(f)
    with open("model_metrics.json") as f:
        metrics = json.load(f)
    with open("feature_ranges.json") as f:
        ranges = json.load(f)
    return model, scaler, feature_names, metrics, ranges


@st.cache_data
def load_data():
    return pd.read_csv("India_GDP_Dataset.csv")


try:
    model, scaler, feature_names, metrics, ranges = load_artifacts()
    artifacts_ok = True
except FileNotFoundError as e:
    artifacts_ok = False
    missing_file = str(e)

df = load_data()

# Human-readable labels & units for engineered/raw features
FEATURE_META = {
    "Inflation_Rate": ("Inflation Rate", "%"),
    "Unemployment_Rate": ("Unemployment Rate", "%"),
    "Exports_USD_Billion": ("Exports", "USD Billion"),
    "Imports_USD_Billion": ("Imports", "USD Billion"),
    "FDI_USD_Billion": ("FDI", "USD Billion"),
    "Industrial_Growth_Rate": ("Industrial Growth Rate", "%"),
    "Services_Growth_Rate": ("Services Growth Rate", "%"),
    "Agriculture_Growth_Rate": ("Agriculture Growth Rate", "%"),
    "Government_Expenditure_PctGDP": ("Govt. Expenditure (% of GDP)", "%"),
    "Fiscal_Deficit_PctGDP": ("Fiscal Deficit (% of GDP)", "%"),
    "Interest_Rate": ("Interest Rate", "%"),
    "Exchange_Rate_INR_USD": ("Exchange Rate (INR/USD)", "₹"),
    "Energy_Consumption_TWh": ("Energy Consumption", "TWh"),
    "Population_Million": ("Population", "Million"),
    "Trade_Balance": ("Trade Balance (Exports - Imports)", "USD Billion"),
    "Export_Import_Ratio": ("Export/Import Ratio", "ratio"),
    "Avg_Sector_Growth": ("Avg. Sector Growth (Industry/Services/Agri)", "%"),
}

ENGINEERED = {"Trade_Balance", "Export_Import_Ratio", "Avg_Sector_Growth"}


# ────────────────────────────────────────────────────────────
# Sidebar
# ────────────────────────────────────────────────────────────
st.sidebar.title("🇮🇳 GDP Growth Predictor")
st.sidebar.caption("ML Regression Pipeline · Streamlit Dashboard")

page = st.sidebar.radio(
    "Navigate",
    ["🔮 Predict", "📊 Dataset Explorer", "📈 Model Performance", "ℹ️ About"],
)

st.sidebar.markdown("---")
if artifacts_ok:
    st.sidebar.success(f"Model loaded: **{metrics['model_name']}**")
    st.sidebar.metric("Test R²", f"{metrics['R2']:.4f}")
else:
    st.sidebar.error(f"Missing artifact file: {missing_file}")
    st.sidebar.info(
        "Run the training notebook first to generate:\n"
        "`best_model.pkl`, `scaler.pkl`, `feature_names.json`, "
        "`model_metrics.json`, `feature_ranges.json`."
    )


# ────────────────────────────────────────────────────────────
# Helper: build engineered feature vector from raw inputs
# ────────────────────────────────────────────────────────────
def build_feature_vector(raw_inputs: dict, feature_names: list) -> pd.DataFrame:
    row = dict(raw_inputs)

    if "Trade_Balance" in feature_names:
        row["Trade_Balance"] = row.get("Exports_USD_Billion", 0) - row.get("Imports_USD_Billion", 0)
    if "Export_Import_Ratio" in feature_names:
        row["Export_Import_Ratio"] = row.get("Exports_USD_Billion", 0) / (row.get("Imports_USD_Billion", 0) + 1e-9)
    if "Avg_Sector_Growth" in feature_names:
        sector_cols = [c for c in ["Industrial_Growth_Rate", "Services_Growth_Rate", "Agriculture_Growth_Rate"] if c in row]
        row["Avg_Sector_Growth"] = np.mean([row[c] for c in sector_cols]) if sector_cols else 0

    ordered = {f: row[f] for f in feature_names}
    return pd.DataFrame([ordered])


# ════════════════════════════════════════════════════════════
# PAGE 1 — PREDICT
# ════════════════════════════════════════════════════════════
if page == "🔮 Predict":
    st.title("🔮 Predict India's GDP Growth Rate")
    st.markdown(
        "Adjust the macroeconomic indicators below and click **Predict** "
        "to estimate India's GDP growth rate (%) using the trained "
        f"**{metrics['model_name'] if artifacts_ok else 'ML'}** model."
    )
    st.markdown("---")

    if not artifacts_ok:
        st.warning("Model artifacts not found — predictions are disabled. See sidebar for details.")
    else:
        # Only ask for raw inputs needed (avoid asking for engineered features twice)
        raw_features = [f for f in feature_names if f not in ENGINEERED]
        # Make sure components of engineered features are also requested even if dropped from final feature list
        extra_needed = set()
        if "Trade_Balance" in feature_names or "Export_Import_Ratio" in feature_names:
            extra_needed.update(["Exports_USD_Billion", "Imports_USD_Billion"])
        if "Avg_Sector_Growth" in feature_names:
            extra_needed.update(["Industrial_Growth_Rate", "Services_Growth_Rate", "Agriculture_Growth_Rate"])

        all_inputs = list(dict.fromkeys(raw_features + [f for f in extra_needed if f not in raw_features]))

        st.subheader("📥 Input Indicators")
        cols = st.columns(3)
        user_inputs = {}

        for i, feat in enumerate(all_inputs):
            label, unit = FEATURE_META.get(feat, (feat.replace("_", " "), ""))
            r = ranges.get(feat, {"min": float(df[feat].min()), "max": float(df[feat].max()), "mean": float(df[feat].mean())})
            col = cols[i % 3]
            with col:
                user_inputs[feat] = st.slider(
                    f"{label} ({unit})" if unit else label,
                    min_value=float(round(r["min"], 2)),
                    max_value=float(round(r["max"], 2)),
                    value=float(round(r["mean"], 2)),
                    step=round((r["max"] - r["min"]) / 100, 4) if r["max"] > r["min"] else 0.01,
                )

        st.markdown("---")
        predict_clicked = st.button("🚀 Predict GDP Growth Rate", use_container_width=True)

        if predict_clicked:
            X_input = build_feature_vector(user_inputs, feature_names)
            X_scaled = scaler.transform(X_input)
            prediction = model.predict(X_scaled)[0]

            st.markdown("### 🎯 Prediction Result")
            res_col1, res_col2, res_col3 = st.columns([1, 1, 1])

            with res_col1:
                st.metric("Predicted GDP Growth Rate", f"{prediction:.2f} %")
            with res_col2:
                avg_growth = df["GDP_Growth_Rate"].mean()
                delta = prediction - avg_growth
                st.metric("vs. Historical Average", f"{avg_growth:.2f} %", delta=f"{delta:+.2f} %")
            with res_col3:
                pct_rank = (df["GDP_Growth_Rate"] < prediction).mean() * 100
                st.metric("Percentile vs. Historical Data", f"{pct_rank:.1f}%")

            # Gauge chart
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=prediction,
                delta={'reference': avg_growth, 'increasing': {'color': "#10b981"}, 'decreasing': {'color': "#ef4444"}},
                title={'text': "Predicted GDP Growth Rate (%)", 'font': {'color': '#e0e0ff'}},
                gauge={
                    'axis': {'range': [df["GDP_Growth_Rate"].min() - 0.5, df["GDP_Growth_Rate"].max() + 0.5],
                             'tickcolor': '#aaaacc'},
                    'bar': {'color': "#7c3aed"},
                    'bgcolor': "#1a1a2e",
                    'borderwidth': 1,
                    'bordercolor': "#2a2a4a",
                    'steps': [
                        {'range': [df["GDP_Growth_Rate"].min() - 0.5, avg_growth], 'color': '#1a1a2e'},
                        {'range': [avg_growth, df["GDP_Growth_Rate"].max() + 0.5], 'color': '#2a2a4a'},
                    ],
                    'threshold': {
                        'line': {'color': "#f59e0b", 'width': 4},
                        'thickness': 0.8,
                        'value': avg_growth,
                    },
                },
                number={'suffix': "%", 'font': {'color': '#a78bfa'}},
            ))
            fig_gauge.update_layout(
                paper_bgcolor="#0f0f1a",
                font={'color': "#e0e0ff"},
                height=350,
                margin=dict(l=30, r=30, t=60, b=10),
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

            with st.expander("🔍 Show input feature vector sent to the model"):
                st.dataframe(X_input.T.rename(columns={0: "Value"}), use_container_width=True)


# ════════════════════════════════════════════════════════════
# PAGE 2 — DATASET EXPLORER
# ════════════════════════════════════════════════════════════
elif page == "📊 Dataset Explorer":
    st.title("📊 Dataset Explorer")
    st.markdown(f"**Shape:** {df.shape[0]:,} rows × {df.shape[1]} columns")

    st.dataframe(df.head(50), use_container_width=True)

    st.markdown("---")
    st.subheader("Distribution of a Selected Feature")
    col_choice = st.selectbox("Select a feature", df.columns, index=len(df.columns) - 1)

    fig_hist = px.histogram(
        df, x=col_choice, nbins=50,
        color_discrete_sequence=["#7c3aed"],
        marginal="box",
    )
    fig_hist.update_layout(
        paper_bgcolor="#0f0f1a", plot_bgcolor="#1a1a2e",
        font_color="#e0e0ff", title=f"Distribution of {col_choice}",
        bargap=0.05,
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    st.markdown("---")
    st.subheader("Correlation with GDP Growth Rate")
    target_corr = df.corr(numeric_only=True)["GDP_Growth_Rate"].drop("GDP_Growth_Rate").sort_values()
    colors = ["#ef4444" if v < 0 else "#10b981" for v in target_corr.values]

    fig_corr = go.Figure(go.Bar(
        x=target_corr.values, y=target_corr.index, orientation="h",
        marker_color=colors,
    ))
    fig_corr.update_layout(
        paper_bgcolor="#0f0f1a", plot_bgcolor="#1a1a2e",
        font_color="#e0e0ff", title="Pearson Correlation with GDP Growth Rate",
        xaxis_title="Correlation Coefficient", height=500,
    )
    st.plotly_chart(fig_corr, use_container_width=True)

    st.markdown("---")
    st.subheader("Full Correlation Heatmap")
    corr_matrix = df.corr(numeric_only=True)
    fig_heat = px.imshow(
        corr_matrix, text_auto=".2f", color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1, aspect="auto",
    )
    fig_heat.update_layout(
        paper_bgcolor="#0f0f1a", font_color="#e0e0ff", height=600,
    )
    st.plotly_chart(fig_heat, use_container_width=True)


# ════════════════════════════════════════════════════════════
# PAGE 3 — MODEL PERFORMANCE
# ════════════════════════════════════════════════════════════
elif page == "📈 Model Performance":
    st.title("📈 Model Performance")

    if not artifacts_ok:
        st.warning("Model metrics not found. Train the model via the notebook first.")
    else:
        st.subheader(f"🏆 Best Model: {metrics['model_name']}")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("R² (Test)", f"{metrics['R2']:.4f}")
        m2.metric("MAE", f"{metrics['MAE']:.4f}")
        m3.metric("RMSE", f"{metrics['RMSE']:.4f}")
        m4.metric("CV R² (5-fold)", f"{metrics['CV_R2']:.4f}")

        st.markdown("---")
        st.subheader("Final Features Used by the Model")
        st.write(", ".join(metrics["features"]))

        st.subheader("Columns Dropped (Low Correlation, |r| < 0.05)")
        st.write(", ".join(metrics["dropped_columns"]) if metrics["dropped_columns"] else "None")

        if hasattr(model, "coef_"):
            st.markdown("---")
            st.subheader("Feature Coefficients")
            coef_df = pd.DataFrame({
                "Feature": metrics["features"],
                "Coefficient": model.coef_,
            }).sort_values("Coefficient")
            colors = ["#ef4444" if v < 0 else "#10b981" for v in coef_df["Coefficient"]]
            fig_coef = go.Figure(go.Bar(
                x=coef_df["Coefficient"], y=coef_df["Feature"], orientation="h",
                marker_color=colors,
            ))
            fig_coef.update_layout(
                paper_bgcolor="#0f0f1a", plot_bgcolor="#1a1a2e",
                font_color="#e0e0ff", title="Model Coefficients (scaled features)",
                height=450,
            )
            st.plotly_chart(fig_coef, use_container_width=True)
        elif hasattr(model, "feature_importances_"):
            st.markdown("---")
            st.subheader("Feature Importance")
            imp_df = pd.DataFrame({
                "Feature": metrics["features"],
                "Importance": model.feature_importances_,
            }).sort_values("Importance")
            fig_imp = go.Figure(go.Bar(
                x=imp_df["Importance"], y=imp_df["Feature"], orientation="h",
                marker_color="#06b6d4",
            ))
            fig_imp.update_layout(
                paper_bgcolor="#0f0f1a", plot_bgcolor="#1a1a2e",
                font_color="#e0e0ff", title="Feature Importance", height=450,
            )
            st.plotly_chart(fig_imp, use_container_width=True)


# ════════════════════════════════════════════════════════════
# PAGE 4 — ABOUT
# ════════════════════════════════════════════════════════════
else:
    st.title("ℹ️ About this App")
    st.markdown("""
This dashboard deploys an **end-to-end ML regression pipeline** that predicts
India's **GDP Growth Rate (%)** from key macroeconomic indicators.

**Pipeline summary:**
1. Loaded the India GDP dataset (10,000 rows × 16 columns)
2. Performed EDA — distributions, boxplots, correlation analysis
3. Dropped low-correlation columns (|r| < 0.05)
4. Engineered new features (e.g. average sector growth)
5. Train/test split (80/20) with `StandardScaler`
6. Trained and compared 6 regression models with 5-fold cross-validation
7. Selected the best model based on test R² and saved it for deployment

**Pages:**
- 🔮 **Predict** — interactively adjust indicators and get a live prediction
- 📊 **Dataset Explorer** — explore distributions and correlations
- 📈 **Model Performance** — view metrics, coefficients/importances

---
*Built with Streamlit, scikit-learn, XGBoost, and Plotly.*
""")
