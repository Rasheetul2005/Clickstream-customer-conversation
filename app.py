from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Customer Conversion Analysis", page_icon="🛍️", layout="wide")
st.title("🛍️ Customer Conversion Analysis")
st.caption("Clickstream-based ML dashboard")

DATA = Path("data")
MODELS = Path("models")

if not (DATA/"session_features.csv").exists():
    st.error("Run: python src/01_download_and_prepare.py")
    st.stop()

df = pd.read_csv(DATA/"session_features.csv")

overview, conversion, revenue, segmentation, performance = st.tabs([
    "📊 Overview","🎯 Conversion","💰 Revenue","👥 Segmentation","📈 Performance"
])

with overview:
    a,b,c,d = st.columns(4)
    a.metric("Sessions", f"{len(df):,}")
    b.metric("Avg clicks", f"{df.click_count.mean():.1f}")
    c.metric("Avg price", f"${df.avg_price.mean():.2f}")
    d.metric("Countries", f"{df.country.nunique():,}")
    st.subheader("Click depth distribution")
    st.bar_chart(df["click_count"].value_counts().sort_index().head(30))
    st.subheader("Main category")
    st.bar_chart(df["main_category"].value_counts())

def model_input(row):
    return row.drop(columns=["session_id","converted_proxy","potential_revenue_proxy"], errors="ignore")

with conversion:
    st.subheader("Customer conversion / high-intent prediction")
    model_file = MODELS/"random_forest_classifier.joblib"
    if model_file.exists():
        model = joblib.load(model_file)
        idx = st.selectbox("Choose session", df.index, key="cls")
        row = df.loc[[idx]]
        pred = int(model.predict(model_input(row))[0])
        prob = float(model.predict_proba(model_input(row))[0,1])
        st.metric("Prediction", "High intent" if pred else "Lower intent")
        st.metric("Probability", f"{prob:.1%}")
        st.info("This is a behavioral proxy target because the raw UCI dataset does not contain an explicit purchase-completion label.")
    else:
        st.warning("Run model training first.")

with revenue:
    st.subheader("Potential revenue estimation")
    model_file = MODELS/"random_forest_regressor.joblib"
    if model_file.exists():
        model = joblib.load(model_file)
        idx = st.selectbox("Choose session", df.index, key="reg")
        row = df.loc[[idx]]
        value = float(model.predict(model_input(row))[0])
        st.metric("Estimated potential value", f"${value:,.2f}")
        st.info("This is a derived potential-value proxy, not observed transaction revenue.")
    else:
        st.warning("Run model training first.")

with segmentation:
    st.subheader("Customer/session segmentation")
    model_file = MODELS/"kmeans.joblib"
    if model_file.exists():
        scaler, kmeans, cols = joblib.load(model_file)
        Z = df[cols].fillna(df[cols].median())
        labels = kmeans.predict(scaler.transform(Z))
        counts = pd.Series(labels).value_counts().sort_index()
        st.bar_chart(counts)
        summary = df.assign(cluster=labels).groupby("cluster")[cols].mean().round(2)
        st.dataframe(summary, use_container_width=True)
    else:
        st.warning("Run model training first.")

with performance:
    st.subheader("Model evaluation")
    cfile = DATA/"classification_results.csv"
    rfile = DATA/"regression_results.csv"
    kfile = DATA/"clustering_results.csv"
    if cfile.exists():
        st.write("Classification")
        st.dataframe(pd.read_csv(cfile), use_container_width=True)
    if rfile.exists():
        st.write("Regression")
        st.dataframe(pd.read_csv(rfile), use_container_width=True)
    if kfile.exists():
        st.write("Clustering")
        st.dataframe(pd.read_csv(kfile), use_container_width=True)
