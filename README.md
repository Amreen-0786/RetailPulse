# RetailPulse — Customer Behavior Analytics & Revenue Intelligence

A runnable educational project based on the Code A Nova Project 4 brief.

## Included
- 15,000-row synthetic retail transaction dataset (not real customer data)
- Data validation and cleaning
- EDA-ready data and business KPIs
- RFM scoring and K-Means segmentation with elbow/inertia and silhouette metrics
- Random Forest churn classifier with held-out ROC-AUC
- SARIMAX 3-month revenue forecast with confidence intervals (linear trend fallback if SARIMAX fails)
- Monthly cohort retention matrix
- Streamlit dashboard with Overview, Segments, Churn, Forecast and Cohorts tabs
- Optional MLflow tracking in `train.py`

## Requirements
Python 3.10+ recommended. The brief specifies Python 3.12.

## Windows / VS Code setup
Open this folder in VS Code. In the integrated PowerShell terminal:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.venv\Scripts\Activate.ps1
```

## Run the dashboard
```powershell
streamlit run app.py
```
If the `streamlit` command is not found:
```powershell
python -m streamlit run app.py
```

## Train and save model artifacts
In a second terminal with the environment activated:
```powershell
python train.py
```
This saves model files and CSV analysis outputs under `models/` and `reports/`.

## Important methodology notes
- The supplied dataset is synthetic and intended for demonstrating the pipeline. Replace it with an approved real retail dataset for a formal internship submission.
- Churn is operationally defined as no purchase in the final 90 days of the available transaction period. This is a proxy label, not verified business churn.
- The churn ROC-AUC is calculated on a stratified held-out test set. Do not claim it exceeds 0.80 unless your run actually reports that.
- The forecast uses SARIMAX(1,1,1) if available and successful; a transparent linear-trend fallback is used otherwise.
- SHAP is listed as a dependency for explainability extensions; the dashboard currently shows Random Forest feature importance, not SHAP explanations.
- MLflow tracking is optional and runs only if local tracking configuration works.

## Suggested next improvements before final submission
1. Add separate reproducible Jupyter notebooks for EDA, feature engineering, segmentation, churn, forecasting and cohort analysis.
2. Add SHAP beeswarm/waterfall plots and explain model limitations.
3. Add statistical hypothesis tests (t-test / chi-square) with null hypotheses and p-values.
4. Deploy the dashboard to Streamlit Community Cloud and write an executive PDF report with evidence-based recommendations.
