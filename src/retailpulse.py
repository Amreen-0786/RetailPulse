"""Core analytics functions for RetailPulse."""
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline

REQUIRED = ["TransactionID","CustomerID","InvoiceDate","Product","Category","Quantity","UnitPrice","Discount","Revenue","Region","Channel"]

def load_data(path):
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    for col in ["Quantity","UnitPrice","Discount","Revenue"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["CustomerID","InvoiceDate","Revenue"])
    df = df.drop_duplicates(subset=["TransactionID"])
    df = df[df["Revenue"] >= 0].copy()
    return df

def build_rfm(df):
    snapshot = df["InvoiceDate"].max() + pd.Timedelta(days=1)
    rfm = df.groupby("CustomerID").agg(
        Recency=("InvoiceDate", lambda x: (snapshot-x.max()).days),
        Frequency=("TransactionID","nunique"),
        Monetary=("Revenue","sum")
    ).reset_index()
    # Quantile scores are descriptive scores; KMeans uses raw standardized RFM values.
    for col, ascending in [("Recency",False),("Frequency",True),("Monetary",True)]:
        rfm[col+"_Score"] = pd.qcut(rfm[col].rank(method="first"), 4, labels=[1,2,3,4] if ascending else [4,3,2,1]).astype(int)
    return rfm

def segment_customers(rfm):
    features = ["Recency","Frequency","Monetary"]
    scaler = StandardScaler()
    X = scaler.fit_transform(rfm[features])
    scores = []
    max_k = min(6, max(2, len(rfm)-1))
    for k in range(2, max_k+1):
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = model.fit_predict(X)
        from sklearn.metrics import silhouette_score
        scores.append((k, float(model.inertia_), float(silhouette_score(X, labels))))
    best_k = max(scores, key=lambda x: x[2])[0]
    model = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    rfm = rfm.copy()
    rfm["Cluster"] = model.fit_predict(X)
    return rfm, pd.DataFrame(scores, columns=["k","inertia","silhouette"]), model, scaler

def build_churn_model(df, rfm):
    # Operational definition: customers with no purchase in the final 90 days of the dataset are labelled churned.
    cutoff = df["InvoiceDate"].max() - pd.Timedelta(days=90)
    last_purchase = df.groupby("CustomerID")["InvoiceDate"].max()
    labels = (last_purchase < cutoff).astype(int).rename("Churn")
    features = rfm.set_index("CustomerID")[["Recency","Frequency","Monetary"]].join(labels).dropna()
    X, y = features[["Recency","Frequency","Monetary"]], features["Churn"]
    if y.nunique() < 2 or y.value_counts().min() < 2:
        raise ValueError("Not enough examples in both churn classes to train a classifier.")
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.25,random_state=42,stratify=y)
    model = Pipeline([("scale",StandardScaler()),("classifier",RandomForestClassifier(n_estimators=250,random_state=42,class_weight="balanced"))])
    model.fit(X_train,y_train)
    probabilities = model.predict_proba(X_test)[:,1]
    auc = float(roc_auc_score(y_test, probabilities))
    return model, X_test, y_test, probabilities, auc, features.reset_index()

def monthly_revenue_forecast(df, periods=3):
    monthly = df.set_index("InvoiceDate").resample("MS")["Revenue"].sum().asfreq("MS", fill_value=0)
    # SARIMAX as requested; fallback to a simple trend if statsmodels is unavailable.
    try:
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        fit = SARIMAX(monthly, order=(1,1,1), seasonal_order=(0,0,0,0), enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
        pred = fit.get_forecast(steps=periods)
        fc = pred.summary_frame()
        result = pd.DataFrame({"Date":fc.index, "Forecast":fc["mean"].values, "Lower":fc["mean_ci_lower"].values, "Upper":fc["mean_ci_upper"].values})
        return monthly, result, "SARIMAX(1,1,1)"
    except Exception as exc:
        x = np.arange(len(monthly))
        slope, intercept = np.polyfit(x, monthly.values, 1)
        future_x = np.arange(len(monthly), len(monthly)+periods)
        vals = np.maximum(0, slope*future_x+intercept)
        result = pd.DataFrame({"Date":pd.date_range(monthly.index[-1]+pd.offsets.MonthBegin(1), periods=periods, freq="MS"), "Forecast":vals, "Lower":vals*.85, "Upper":vals*1.15})
        return monthly, result, f"Linear trend fallback ({type(exc).__name__})"

def cohort_retention(df):
    data = df[["CustomerID","InvoiceDate","Revenue"]].copy()
    data["OrderMonth"] = data["InvoiceDate"].dt.to_period("M")
    data["CohortMonth"] = data.groupby("CustomerID")["InvoiceDate"].transform("min").dt.to_period("M")
    data["CohortIndex"] = (data["OrderMonth"].dt.year-data["CohortMonth"].dt.year)*12 + (data["OrderMonth"].dt.month-data["CohortMonth"].dt.month) + 1
    counts = data.groupby(["CohortMonth","CohortIndex"])["CustomerID"].nunique().unstack()
    retention = counts.divide(counts.iloc[:,0], axis=0)
    retention.index = retention.index.astype(str)
    return retention

def business_summary(df, rfm, auc, forecast):
    return {
        "transactions": int(len(df)),
        "customers": int(df["CustomerID"].nunique()),
        "revenue": float(df["Revenue"].sum()),
        "average_order_value": float(df.groupby("TransactionID")["Revenue"].sum().mean()),
        "churn_auc": float(auc),
        "next_3_month_forecast": float(forecast["Forecast"].sum()),
        "top_region": str(df.groupby("Region")["Revenue"].sum().idxmax()),
        "top_category": str(df.groupby("Category")["Revenue"].sum().idxmax())
    }
