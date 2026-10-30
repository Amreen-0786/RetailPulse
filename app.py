from pathlib import Path
import sys
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT/"src"))
import streamlit as st
import pandas as pd
import plotly.express as px
from retailpulse import load_data, build_rfm, segment_customers, build_churn_model, monthly_revenue_forecast, cohort_retention, business_summary

st.set_page_config(page_title="RetailPulse | Retail Intelligence", page_icon="📊", layout="wide")
st.title("📊 RetailPulse")
st.caption("Customer Behavior Analytics & Revenue Intelligence Platform")
@st.cache_data
def get_data(path):
    return load_data(path)

data_path = ROOT/"data"/"retail_transactions.csv"
try:
    df = get_data(str(data_path))
except Exception as e:
    st.error(f"Could not load dataset: {e}")
    st.stop()

with st.spinner("Running customer analytics and machine-learning models..."):
    rfm = build_rfm(df)
    segmented, k_scores, kmeans, scaler = segment_customers(rfm)
    churn_model, X_test, y_test, probs, auc, churn_features = build_churn_model(df, rfm)
    monthly, forecast, forecast_method = monthly_revenue_forecast(df)
    retention = cohort_retention(df)
    summary = business_summary(df, rfm, auc, forecast)

c1,c2,c3,c4 = st.columns(4)
c1.metric("Transactions", f'{summary["transactions"]:,}')
c2.metric("Customers", f'{summary["customers"]:,}')
c3.metric("Total revenue", f'${summary["revenue"]:,.0f}')
c4.metric("Churn ROC-AUC", f'{auc:.3f}')

tabs = st.tabs(["Overview","Segments","Churn","Forecast","Cohorts"])
with tabs[0]:
    st.subheader("Business overview")
    left,right = st.columns(2)
    with left:
        monthly_plot = monthly.rename("Revenue").reset_index()
        st.plotly_chart(px.line(monthly_plot,x="InvoiceDate",y="Revenue",title="Monthly revenue trend",markers=True), use_container_width=True)
    with right:
        category = df.groupby("Category",as_index=False)["Revenue"].sum().sort_values("Revenue",ascending=False)
        st.plotly_chart(px.bar(category,x="Category",y="Revenue",title="Revenue by category"), use_container_width=True)
    st.write(f'**Top region:** {summary["top_region"]} · **Top category:** {summary["top_category"]} · **Average order value:** ${summary["average_order_value"]:,.2f}')
    st.info("Churn label definition: a customer is labelled churned when their last purchase is more than 90 days before the dataset's final transaction date. This is a proxy label derived from transaction history.")
with tabs[1]:
    st.subheader("RFM customer segmentation")
    st.plotly_chart(px.scatter(segmented,x="Recency",y="Monetary",color=segmented["Cluster"].astype(str),size="Frequency",hover_data=["CustomerID"],title="Customer clusters (RFM)"),use_container_width=True)
    st.dataframe(segmented.groupby("Cluster")[["Recency","Frequency","Monetary"]].mean().round(2))
    st.plotly_chart(px.line(k_scores,x="k",y="silhouette",markers=True,title="Silhouette score by K (higher is better)"),use_container_width=True)
with tabs[2]:
    st.subheader("Churn prediction")
    st.metric("Held-out test ROC-AUC",f"{auc:.3f}")
    st.caption("AUC is calculated on a stratified 25% hold-out set; results may vary with different data.")
    importance = pd.DataFrame({"Feature":["Recency","Frequency","Monetary"],"Importance":churn_model.named_steps["classifier"].feature_importances_})
    st.plotly_chart(px.bar(importance,x="Feature",y="Importance",title="Random Forest feature importance"),use_container_width=True)
    st.dataframe(churn_features.head(20))
with tabs[3]:
    st.subheader("Three-month revenue forecast")
    st.caption(f"Model: {forecast_method}")
    hist = monthly.rename("Revenue").reset_index().rename(columns={"InvoiceDate":"Date"})
    hist["Type"]="Historical"
    future = forecast.copy()
    future["Type"]="Forecast"
    combined = pd.concat([hist[["Date","Revenue","Type"]].rename(columns={"Revenue":"Amount"}),future[["Date","Forecast","Type"]].rename(columns={"Forecast":"Amount"})])
    st.plotly_chart(px.line(combined,x="Date",y="Amount",color="Type",markers=True,title="Revenue history and forecast"),use_container_width=True)
    st.dataframe(forecast.style.format({"Forecast":"${:,.2f}","Lower":"${:,.2f}","Upper":"${:,.2f}"}))
with tabs[4]:
    st.subheader("Monthly cohort retention")
    st.plotly_chart(px.imshow(retention,aspect="auto",color_continuous_scale="Blues",labels=dict(x="Months since first purchase",y="Cohort month",color="Retention"),title="Customer retention heatmap"),use_container_width=True)
    st.caption("Each cell is the share of a cohort that made a purchase in that month relative to its first month.")

st.divider()
st.subheader("Executive takeaways")
st.write(f'- Dataset contains {summary["transactions"]:,} transactions from {summary["customers"]:,} customers.')
st.write(f'- Highest-revenue region is **{summary["top_region"]}** and leading category is **{summary["top_category"]}**.')
st.write(f'- The next three months have a combined model forecast of **${summary["next_3_month_forecast"]:,.0f}**.')
st.write("- Use the segment profiles to tailor retention campaigns; prioritize high-recency-risk customers for outreach.")
