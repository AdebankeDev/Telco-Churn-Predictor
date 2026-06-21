# Telco Customer Churn Predictor

🔗 **Live app:** [telco-churn-predictor-v1.streamlit.app](https://telco-churn-predictor-v1.streamlit.app/)

A deployed Streamlit app that predicts customer churn for a telecom provider, using a Logistic Regression model (test ROC-AUC 0.8465). Supports single-customer predictions via a form, batch scoring via CSV upload, and a SHAP explanation showing *why* the model made each call.

## What it does

- **Single Customer** — fill in a customer's details, get a churn probability, a Yes/No risk flag (threshold 0.55), and a SHAP waterfall chart breaking down exactly which factors drove that specific prediction.
- **Batch Upload (CSV)** — upload a CSV of customers, get back the same table with `Churn_Probability` and `Predicted_Churn` columns added, sortable and downloadable.
- **About this model** — the model card: metrics, threshold rationale, and the top churn drivers identified through SHAP analysis.

## Model summary

| Metric | Value |
|---|---|
| Model | Logistic Regression |
| Decision threshold | 0.55 |
| Test accuracy | 0.78 |
| Test recall (churn class) | 0.73 |
| Test precision (churn class) | 0.56 |
| Test ROC-AUC | 0.8465 |

This configuration was chosen after testing three alternatives — hyperparameter tuning, an LR + XGBoost ensemble, and a separately-tuned XGBoost model. All three showed promising results on validation data but underperformed plain LR on the held-out test set, confirming the original configuration was already close to optimal for this feature set.

## Project structure

```
app.py                  → the Streamlit app
feature_engineering.py  → feature engineering (must match training exactly)
requirements.txt        → pinned Python dependencies
churn_model.pkl         → trained scikit-learn pipeline
churn_threshold.pkl     → decision threshold (0.55)
```

## Running locally

1. **Clone this repo** and open it in VS Code.
   ```
   git clone https://github.com/<your-username>/telco-churn-predictor.git
   cd telco-churn-predictor
   ```

2. **Confirm the model files are present.** `churn_model.pkl` and `churn_threshold.pkl` should already be in the repo, alongside `app.py`. If they're missing, the app will load but show an on-screen error explaining what's needed.

3. **Create and activate a virtual environment:**
   ```
   python -m venv .venv
   ```
   - Windows (PowerShell): `.venv\Scripts\Activate.ps1`
   - Windows (cmd): `.venv\Scripts\activate.bat`
   - Mac/Linux: `source .venv/bin/activate`

   VS Code may prompt "Select environment for this workspace" — choose the `.venv` one.

   *(Using `uv` instead of `venv`/`pip`?)*
   ```
   uv venv --python 3.11
   ```
   then activate the same way as above.

4. **Install dependencies:**
   ```
   pip install -r requirements.txt
   ```
   or with `uv`:
   ```
   uv pip install -r requirements.txt
   ```
   Versions are pinned to match exactly what trained the model (scikit-learn 1.6.1, numpy 2.0.2, shap 0.51.0, etc.) — installing different versions risks `joblib.load()` failing or silently loading a mismatched model structure.

5. **Run the app:**
   ```
   streamlit run app.py
   ```
   or with `uv`:
   ```
   uv run streamlit run app.py
   ```
   Opens automatically at `http://localhost:8501`. Leave the terminal running — closing it stops the app.

## Deploying to Streamlit Community Cloud

1. Push this repo to GitHub (the `.pkl` files are small enough, ~10KB each, to include directly).
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in with GitHub, click "New app."
3. Select this repo, branch `main`, and set the main file path to `app.py`.
4. Click Deploy — it builds from `requirements.txt` automatically, no extra config needed.

## Notes

- `feature_engineering.py` mirrors the exact preprocessing and feature engineering used during training, including a frozen `TRAINING_MONTHLY_CHARGES_MEDIAN` constant for the `high_risk` feature. This was a deliberate fix: computing the median live from whatever batch is passed in would make `high_risk` meaningless for single-row predictions (the median of one value is just that value), so the training-set median is frozen as a constant instead, ensuring single and batch predictions behave consistently. If the notebook is retrained, this constant must be updated to match the new training run.
- The in-app SHAP explanation uses a lightweight background reference rather than shipping the full training set, so values are directionally consistent with the training notebook's SHAP analysis but not numerically identical to the last decimal.
- Full model development — including the comparison of all four modeling approaches (threshold tuning, hyperparameter search, ensembling, and XGBoost) and the complete SHAP analysis — is documented in the accompanying Jupyter notebook.
