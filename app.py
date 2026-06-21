"""
Telco Customer Churn Predictor — Streamlit app.

Run locally with:
    streamlit run app.py
"""

import io

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st

from feature_engineering import RAW_COLUMNS, engineer_features

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Churn Predictor",
    page_icon="📉",
    layout="wide",
)

# --- Blue theme palette ---
NAVY = "#0B2545"        # deep navy — headers, primary text accents
BLUE = "#1B6FC9"        # primary action blue — buttons, links, highlights
LIGHT_BLUE = "#E8F1FB"  # soft panel backgrounds
SKY = "#5FA8E0"         # secondary accent
DANGER = "#D64550"      # churn risk red (kept distinct from blue theme on purpose)
SAFE = "#1E8E6E"        # retained / low-risk green
INK = "#1C2B3A"         # body text
MUTED = "#5C7185"       # secondary text
BORDER = "#D7E4F2"

st.markdown(
    f"""
    <style>
    /* ---------- Global ---------- */
    .stApp {{
        background-color: #F4F8FC;
    }}
    html, body, [class*="css"] {{
        color: {INK};
    }}
    h1, h2, h3 {{
        color: {NAVY} !important;
        font-weight: 700 !important;
    }}
    p, label, .stMarkdown {{
        color: {INK};
    }}

    /* ---------- Sidebar ---------- */
    section[data-testid="stSidebar"] {{
        background-color: {NAVY};
    }}
    section[data-testid="stSidebar"] * {{
        color: #E8F1FB !important;
    }}
    section[data-testid="stSidebar"] .stRadio label {{
        font-weight: 500;
    }}
    section[data-testid="stSidebar"] hr {{
        border-color: rgba(255,255,255,0.15);
    }}

    /* ---------- Buttons ---------- */
    .stButton > button, .stFormSubmitButton > button {{
        background-color: {BLUE} !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        padding: 0.6rem 1.2rem !important;
        transition: background-color 0.15s ease-in-out;
    }}
    .stButton > button:hover, .stFormSubmitButton > button:hover {{
        background-color: {NAVY} !important;
    }}
    .stDownloadButton > button {{
        background-color: {NAVY} !important;
        color: white !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
    }}

    /* ---------- Cards ---------- */
    .metric-card {{
        border: 1px solid {BORDER};
        border-radius: 12px;
        padding: 1.1rem 1.3rem;
        background-color: white;
        box-shadow: 0 1px 3px rgba(11, 37, 69, 0.06);
    }}
    .metric-card b {{
        font-size: 1.6rem;
        color: {NAVY};
        display: block;
        margin-bottom: 0.2rem;
    }}

    /* ---------- Risk badge ---------- */
    .churn-badge {{
        display: inline-block;
        padding: 0.45rem 1.1rem;
        border-radius: 999px;
        font-weight: 700;
        font-size: 1rem;
        letter-spacing: 0.01em;
    }}
    .badge-high {{ background-color: {DANGER}1A; color: {DANGER}; border: 1px solid {DANGER}40; }}
    .badge-low  {{ background-color: {SAFE}1A; color: {SAFE}; border: 1px solid {SAFE}40; }}

    /* ---------- Section dividers ---------- */
    hr {{
        border-color: {BORDER};
    }}

    /* ---------- Form section headers ---------- */
    .form-section-label {{
        color: {BLUE};
        font-weight: 700;
        font-size: 0.95rem;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-bottom: 0.5rem;
        border-bottom: 2px solid {LIGHT_BLUE};
        padding-bottom: 0.4rem;
    }}

    /* ---------- Inputs ---------- */
    .stSelectbox > div > div, .stNumberInput > div > div {{
        border-radius: 8px !important;
        border-color: {BORDER} !important;
    }}

    /* ---------- Hero banner ---------- */
    .hero-banner {{
        background: linear-gradient(135deg, {NAVY} 0%, {BLUE} 100%);
        border-radius: 14px;
        padding: 1.8rem 2rem;
        margin-bottom: 1.5rem;
        color: white;
    }}
    .hero-banner h1 {{
        color: white !important;
        margin-bottom: 0.3rem;
    }}
    .hero-banner p {{
        color: #DCEBFB;
        margin: 0;
        font-size: 1.02rem;
    }}

    /* ---------- Dataframe ---------- */
    [data-testid="stDataFrame"] {{
        border: 1px solid {BORDER};
        border-radius: 10px;
        overflow: hidden;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Cached resources
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model_and_threshold():
    model = joblib.load("churn_model.pkl")
    threshold = joblib.load("churn_threshold.pkl")
    return model, threshold


@st.cache_resource
def load_explainer(_model):
    """
    Build a SHAP LinearExplainer once and cache it.
    Uses a small synthetic background since we don't ship the original
    training data with the app — the model's own coefficients dominate
    the explanation, so this is a lightweight, fast approximation.
    """
    classifier = _model.named_steps["classifier"]
    preprocessor = _model[:-1]
    return classifier, preprocessor


try:
    model, threshold = load_model_and_threshold()
    classifier, preprocessor = load_explainer(model)
    MODEL_LOADED = True
except FileNotFoundError:
    MODEL_LOADED = False


# ---------------------------------------------------------------------------
# Prediction helpers
# ---------------------------------------------------------------------------
def predict(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Run the full pipeline on raw customer rows and return predictions."""
    df_features = engineer_features(df_raw)
    probs = model.predict_proba(df_features)[:, 1]
    preds = (probs >= threshold).astype(int)
    result = df_raw.copy()
    result["Churn_Probability"] = probs.round(4)
    result["Predicted_Churn"] = np.where(preds == 1, "Yes", "No")
    return result, df_features


def explain_single(df_features_row: pd.DataFrame, background_X: pd.DataFrame = None):
    """Generate a SHAP waterfall explanation for one customer."""
    X_transformed = preprocessor.transform(df_features_row)
    if hasattr(X_transformed, "toarray"):
        X_transformed = X_transformed.toarray()

    feature_names = preprocessor.named_steps["preprocessor"].get_feature_names_out()

    # Background: reuse the single row scaled near zero isn't ideal, but
    # without shipping the full training set, a zero-vector background
    # (post-scaling "average-ish" reference) keeps this fast and self-contained.
    background = np.zeros((1, X_transformed.shape[1]))
    explainer = shap.LinearExplainer(classifier, background)
    shap_values = explainer(X_transformed)
    shap_values.feature_names = list(feature_names)
    return shap_values


# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
st.sidebar.markdown("## 📉 Churn Predictor")
st.sidebar.caption("Logistic Regression · threshold 0.55 · test AUC 0.85")
st.sidebar.markdown("<hr>", unsafe_allow_html=True)
page = st.sidebar.radio("Mode", ["Single Customer", "Batch Upload (CSV)", "About this model"])

if not MODEL_LOADED:
    st.error(
        "Couldn't find **churn_model.pkl** and **churn_threshold.pkl** in the app folder. "
        "Place both files alongside `app.py` and refresh."
    )
    st.stop()


# ---------------------------------------------------------------------------
# Page: Single Customer
# ---------------------------------------------------------------------------
if page == "Single Customer":
    st.markdown(
        """
        <div class="hero-banner">
            <h1>Predict churn for a single customer</h1>
            <p>Fill in the customer's details below, then run the prediction to see their churn risk and what's driving it.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.form("single_customer_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown('<div class="form-section-label">Demographics</div>', unsafe_allow_html=True)
            gender = st.selectbox("Gender", ["Female", "Male"])
            senior = st.selectbox("Senior citizen", ["No", "Yes"])
            partner = st.selectbox("Has partner", ["No", "Yes"])
            dependents = st.selectbox("Has dependents", ["No", "Yes"])
            tenure = st.number_input("Tenure (months)", min_value=0, max_value=72, value=12)

        with col2:
            st.markdown('<div class="form-section-label">Services</div>', unsafe_allow_html=True)
            phone = st.selectbox("Phone service", ["Yes", "No"])
            multiple_lines = st.selectbox("Multiple lines", ["No", "Yes", "No phone service"])
            internet = st.selectbox("Internet service", ["DSL", "Fiber optic", "No"])
            online_security = st.selectbox("Online security", ["No", "Yes", "No internet service"])
            online_backup = st.selectbox("Online backup", ["No", "Yes", "No internet service"])
            device_protection = st.selectbox("Device protection", ["No", "Yes", "No internet service"])
            tech_support = st.selectbox("Tech support", ["No", "Yes", "No internet service"])
            streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
            streaming_movies = st.selectbox("Streaming movies", ["No", "Yes", "No internet service"])

        with col3:
            st.markdown('<div class="form-section-label">Account &amp; billing</div>', unsafe_allow_html=True)
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
            paperless = st.selectbox("Paperless billing", ["Yes", "No"])
            payment = st.selectbox(
                "Payment method",
                ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
            )
            monthly_charges = st.number_input("Monthly charges ($)", min_value=0.0, value=70.0, step=0.5)
            total_charges = st.number_input(
                "Total charges ($)", min_value=0.0, value=float(monthly_charges) * max(tenure, 1), step=1.0
            )

        submitted = st.form_submit_button("Predict churn risk", type="primary", use_container_width=True)

    if submitted:
        raw_row = pd.DataFrame([{
            "gender": gender,
            "SeniorCitizen": 1 if senior == "Yes" else 0,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone,
            "MultipleLines": multiple_lines,
            "InternetService": internet,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless,
            "PaymentMethod": payment,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
        }])

        result, df_features = predict(raw_row)
        prob = result["Churn_Probability"].iloc[0]
        pred = result["Predicted_Churn"].iloc[0]

        st.divider()
        res_col1, res_col2 = st.columns([1, 2])

        with res_col1:
            badge_class = "badge-high" if pred == "Yes" else "badge-low"
            badge_text = "⚠️ Likely to churn" if pred == "Yes" else "✓ Likely to stay"
            st.markdown(f'<span class="churn-badge {badge_class}">{badge_text}</span>', unsafe_allow_html=True)
            st.metric("Churn probability", f"{prob:.0%}")
            st.caption(f"Flagged if probability ≥ {threshold:.0%} (model's chosen threshold)")

        with res_col2:
            st.markdown("**Why this prediction?**")
            with st.spinner("Computing SHAP explanation..."):
                shap_values = explain_single(df_features)

            fig, ax = plt.subplots(figsize=(8, 4.5))
            shap.plots.waterfall(shap_values[0], max_display=10, show=False)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

            st.caption(
                "Red bars push the prediction toward churn; blue bars push toward staying. "
                "Bars are ordered by impact, largest at top."
            )


# ---------------------------------------------------------------------------
# Page: Batch Upload
# ---------------------------------------------------------------------------
elif page == "Batch Upload (CSV)":
    st.markdown(
        """
        <div class="hero-banner">
            <h1>Predict churn for multiple customers</h1>
            <p>Upload a CSV with the same columns as the Telco dataset — no <code style="background-color:rgba(255,255,255,0.2); color:#FFFFFF; padding:0.15rem 0.45rem; border-radius:5px; font-weight:600;">customerID</code> or <code style="background-color:rgba(255,255,255,0.2); color:#FFFFFF; padding:0.15rem 0.45rem; border-radius:5px; font-weight:600;">Churn</code> column needed.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("Expected columns"):
        st.code(", ".join(RAW_COLUMNS), language=None)

    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded is not None:
        try:
            raw_df = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"Couldn't read this file: {e}")
            st.stop()

        # Drop columns the model doesn't use, if present
        for extra_col in ["customerID", "Churn"]:
            if extra_col in raw_df.columns:
                raw_df = raw_df.drop(columns=[extra_col])

        missing = set(RAW_COLUMNS) - set(raw_df.columns)
        if missing:
            st.error(f"Missing required columns: {', '.join(sorted(missing))}")
            st.stop()

        with st.spinner(f"Scoring {len(raw_df)} customers..."):
            result, _ = predict(raw_df)

        n_flagged = (result["Predicted_Churn"] == "Yes").sum()
        c1, c2, c3 = st.columns(3)
        c1.markdown(f'<div class="metric-card"><b>{len(result)}</b><br>Customers scored</div>', unsafe_allow_html=True)
        c2.markdown(f'<div class="metric-card"><b>{n_flagged}</b><br>Flagged as high-risk</div>', unsafe_allow_html=True)
        c3.markdown(
            f'<div class="metric-card"><b>{n_flagged / len(result):.0%}</b><br>Of uploaded customers</div>',
            unsafe_allow_html=True,
        )

        st.divider()
        sort_by_risk = st.checkbox("Sort by churn probability (highest first)", value=True)
        display_df = result.sort_values("Churn_Probability", ascending=False) if sort_by_risk else result
        st.dataframe(display_df, use_container_width=True, height=420)

        csv_bytes = display_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download results as CSV",
            data=csv_bytes,
            file_name="churn_predictions.csv",
            mime="text/csv",
            type="primary",
        )


# ---------------------------------------------------------------------------
# Page: About
# ---------------------------------------------------------------------------
else:
    st.markdown(
        """
        <div class="hero-banner">
            <h1>About this model</h1>
            <p>The model card — how it was built, how to read the output, and what drives churn predictions.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        **Model:** Logistic Regression (scikit-learn `Pipeline`: `ColumnTransformer` → `StandardScaler` → `LogisticRegression`)

        **Decision threshold:** 0.55 — chosen to balance recall (catching actual churners) against overall accuracy,
        after testing four approaches (threshold tuning, hyperparameter search, an LR + XGBoost ensemble, and a
        separately-tuned XGBoost model). All three alternatives looked competitive on validation data but underperformed
        plain LR on the held-out test set.

        **Test set performance:**
        | Metric | Value |
        |---|---|
        | Accuracy | 0.78 |
        | Recall (churn class) | 0.73 |
        | Precision (churn class) | 0.56 |
        | ROC-AUC | 0.8465 |

        **How to read the probability:** the model outputs a churn probability for each customer. Anyone at or above
        the 0.55 threshold is flagged "likely to churn." This threshold deliberately favors catching more at-risk
        customers over minimizing false alarms, since for most subscription businesses, the cost of missing a churner
        outweighs the cost of an unnecessary retention offer.

        **Top churn drivers** (from SHAP analysis on the training set): contract type (month-to-month is highest risk),
        internet service type (fiber optic customers churn more), tenure (newer customers are higher risk), and an
        engineered feature, `charges_per_tenure`, which captures customers paying a lot relative to how long they've
        been with the company — found to be the single strongest individual driver for a specific high-risk segment.
        """
    )