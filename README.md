# Churn Predictor — Streamlit App

A web app for the Telco Customer Churn model: predict churn for a single
customer via a form, or batch-score a CSV upload — both with a SHAP
explanation of *why* the model made each call.

## Files

```
app.py                  → the Streamlit app
feature_engineering.py  → shared feature engineering (must match training exactly)
requirements.txt        → Python dependencies
churn_model.pkl         → YOU ADD THIS (from your notebook)
churn_threshold.pkl     → YOU ADD THIS (from your notebook)
```

## Setup (VS Code)

1. **Copy your model files in.** From your churn project, copy
   `churn_model.pkl` and `churn_threshold.pkl` into this same folder, next
   to `app.py`. The app won't start without them.

2. **Open this folder in VS Code.**
   `File → Open Folder...` → select this folder.

3. **Create a virtual environment** (Terminal in VS Code, `Ctrl+\``):
   ```
   python -m venv .venv
   ```
   Activate it:
   - Windows: `.venv\Scripts\activate`
   - Mac/Linux: `source .venv/bin/activate`

   VS Code may prompt "Select environment for this workspace" — choose the
   `.venv` one.

4. **Install dependencies:**
   ```
   pip install -r requirements.txt
   ```

5. **Run the app:**
   ```
   streamlit run app.py
   ```
   This opens automatically in your browser at `http://localhost:8501`.
   Leave the terminal running — closing it stops the app.

## Using the app

- **Single Customer** — fill in the form, click "Predict churn risk." You'll
  get a probability, a Yes/No flag (based on the 0.55 threshold), and a SHAP
  waterfall chart showing which factors drove that specific prediction.

- **Batch Upload (CSV)** — upload a CSV with the same columns as the Telco
  dataset (`customerID` and `Churn` columns are fine to leave in — the app
  drops them automatically). Get back the same table with two new columns:
  `Churn_Probability` and `Predicted_Churn`, sortable and downloadable.

- **About this model** — the model card: metrics, threshold rationale, and
  top churn drivers, pulled straight from your project's analysis.

## Deploying to Streamlit Community Cloud

1. Push this folder to a GitHub repo (include the `.pkl` files — they're
   small enough, ~10KB each).
2. Go to [share.streamlit.io](https://share.streamlit.io), connect your
   GitHub, and point it at this repo with `app.py` as the entry point.
3. It builds from `requirements.txt` automatically — no extra config needed.

## Notes

- `feature_engineering.py` must always match whatever logic produced the
  features your saved model was trained on. If you change the notebook's
  feature engineering later, update this file too, or predictions will be
  wrong.
- The SHAP explanation in the app uses a zero-vector background (rather than
  shipping your full training set), so the waterfall values are a fast,
  self-contained approximation — directionally consistent with your
  notebook's SHAP analysis, but not numerically identical down to the last
  decimal.
