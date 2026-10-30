from pathlib import Path
import sys, joblib
ROOT=Path(__file__).parent
sys.path.insert(0,str(ROOT/"src"))
from retailpulse import load_data, build_rfm, segment_customers, build_churn_model, monthly_revenue_forecast, cohort_retention

df=load_data(ROOT/"data"/"retail_transactions.csv")
rfm=build_rfm(df)
segmented,k_scores,kmeans,scaler=segment_customers(rfm)
model,X_test,y_test,probs,auc,features=build_churn_model(df,rfm)
monthly,forecast,method=monthly_revenue_forecast(df)
(ROOT/"models").mkdir(exist_ok=True)
joblib.dump({"model":model,"kmeans":kmeans,"scaler":scaler},ROOT/"models"/"retailpulse_models.pkl")
segmented.to_csv(ROOT/"reports"/"customer_segments.csv",index=False)
forecast.to_csv(ROOT/"reports"/"revenue_forecast.csv",index=False)
cohort_retention(df).to_csv(ROOT/"reports"/"cohort_retention.csv")
print(f"Training complete. Held-out churn ROC-AUC: {auc:.3f}")
print(f"Forecast method: {method}")
# Optional MLflow tracking. The app still works if MLflow is not installed/configured.
try:
    import mlflow
    with mlflow.start_run(run_name="retailpulse-random-forest"):
        mlflow.log_metric("heldout_roc_auc",auc)
        mlflow.log_param("model","RandomForestClassifier")
        mlflow.log_artifact(str(ROOT/"models"/"retailpulse_models.pkl"))
except Exception as e:
    print(f"MLflow tracking skipped: {e}")
