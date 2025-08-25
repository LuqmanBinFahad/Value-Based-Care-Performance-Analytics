import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
import plotly.express as px

st.set_page_config(page_title="Value-Based Care Dashboard", layout="wide")

@st.cache_data
def load_data():
    patients = pd.read_csv("data/patients.csv")
    encounters = pd.read_csv("data/encounters.csv")
    quality = pd.read_csv("data/quality_measures.csv")
    contracts = pd.read_csv("data/contracts.csv")
    return patients, encounters, quality, contracts

patients, encounters, quality, contracts = load_data()

# Sidebar filters
st.sidebar.header("Filters")
region = st.sidebar.multiselect("Region", options=sorted(patients["region"].unique()), default=list(sorted(patients["region"].unique())))
sex = st.sidebar.multiselect("Sex", options=sorted(patients["sex"].unique()), default=list(sorted(patients["sex"].unique())))
age_min, age_max = st.sidebar.slider("Age range", min_value=int(patients["age"].min()), max_value=int(patients["age"].max()), value=(int(patients["age"].min()), int(patients["age"].max())))

filtered = patients.query("region in @region and sex in @sex and @age_min <= age <= @age_max")
st.sidebar.markdown(f"**Filtered Patients:** {len(filtered):,}")

# ---- Overview KPIs ----
st.title("Value-Based Care Performance Analytics — Prototype")

col1, col2, col3, col4 = st.columns(4)
with col1:
    pmpm = filtered["pmpm_cost"].mean() if len(filtered) else np.nan
    st.metric("Avg PMPM ($)", f"{pmpm:,.2f}" if not np.isnan(pmpm) else "—")
with col2:
    readmit_rate = filtered["readmit_30d"].mean() if len(filtered) else np.nan
    st.metric("30-day Readmission Rate", f"{readmit_rate*100:,.1f}%" if not np.isnan(readmit_rate) else "—")
with col3:
    avg_hcc = filtered["hcc_score"].mean() if len(filtered) else np.nan
    st.metric("Avg HCC Risk", f"{avg_hcc:,.2f}" if not np.isnan(avg_hcc) else "—")
with col4:
    cmor = filtered["comorbidities"].mean() if len(filtered) else np.nan
    st.metric("Avg Comorbidities", f"{cmor:,.2f}" if not np.isnan(cmor) else "—")

st.markdown("---")

# ---- Cost & Utilization Trends ----
st.subheader("Cost & Utilization Trends")
enc_join = encounters.merge(filtered[["patient_id"]], on="patient_id", how="inner")
if len(enc_join):
    monthly = enc_join.groupby("month").agg(total_cost=("total_cost","sum"),
                                            admits=("inpatient_admit","sum"),
                                            ed=("ed_visit","sum"),
                                            op=("outpatient_visits","sum")).reset_index()
    fig1 = px.line(monthly, x="month", y="total_cost", title="Total Cost by Month")
    st.plotly_chart(fig1, use_container_width=True)
    fig2 = px.bar(monthly, x="month", y=["admits","ed","op"], barmode="group", title="Utilization by Month")
    st.plotly_chart(fig2, use_container_width=True)
else:
    st.info("No encounters for current filter.")

st.markdown("---")

# ---- Quality Measures ----
st.subheader("Quality Measures (HEDIS-like)")
m_options = quality["measure_code"].unique().tolist()
m_pick = st.selectbox("Select a measure", options=m_options, index=0)
qf = quality[quality["measure_code"]==m_pick].copy()
qf["rate_pct"] = qf["rate"]*100
figq = px.line(qf, x="month", y="rate_pct", markers=True, title=f"{qf['measure_name'].iloc[0]} — Rate (%)")
st.plotly_chart(figq, use_container_width=True)
st.dataframe(qf[["month","measure_code","measure_name","numerator","denominator","rate_pct"]])

st.markdown("---")

# ---- Risk Stratification ----
st.subheader("Risk Stratification & High-Risk Cohorts")
risk_bins = pd.cut(filtered["hcc_score"], bins=[0,0.75,1.25,2.0,10], labels=["Low","Below Avg","Above Avg","High"])
risk_view = filtered.assign(risk_segment=risk_bins)
dist = risk_view["risk_segment"].value_counts(normalize=True).sort_index()*100
figr = px.bar(x=dist.index.astype(str), y=dist.values, labels={'x':'Risk Segment','y':'% of Patients'}, title="Risk Segment Distribution")
st.plotly_chart(figr, use_container_width=True)

# High-risk cohort table
hi = risk_view.query("risk_segment=='High' or (hcc_score>=1.5 and comorbidities>=3)")
st.caption("High-risk cohort rule: HCC ≥ 1.5 and ≥3 comorbidities (or 'High' segment).")
st.dataframe(hi[["patient_id","age","sex","hcc_score","comorbidities","prev_admissions","length_of_stay","pmpm_cost","readmit_30d"]].sort_values("hcc_score", ascending=False))

# ---- Readmission Prediction ----
st.markdown("---")
st.subheader("Readmission Prediction (Logistic Regression)")
if len(filtered) > 100:
    X = filtered[["age","hcc_score","comorbidities","prev_admissions","length_of_stay"]].copy()
    y = filtered["readmit_30d"].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    clf = LogisticRegression(max_iter=200)
    clf.fit(X_train, y_train)
    y_prob = clf.predict_proba(X_test)[:,1]
    auc = roc_auc_score(y_test, y_prob)
    st.metric("AUC (holdout)", f"{auc:0.3f}")
    # Score all filtered patients
    scores = clf.predict_proba(X)[:,1]
    scored = filtered.assign(readmit_risk=scores)
    st.dataframe(scored.sort_values("readmit_risk", ascending=False).head(25)[["patient_id","age","hcc_score","comorbidities","prev_admissions","length_of_stay","readmit_risk"]])
else:
    st.info("Not enough patients after filtering to train a model.")

# ---- Contract Benchmarking ----
st.markdown("---")
st.subheader("Contract Benchmarking & ROI")
merged = filtered.merge(contracts, on="patient_id", how="left")
bench = merged.groupby("aco_id", dropna=False).agg(
    members=("patient_id","nunique"),
    avg_benchmark=("benchmark_pmpm","mean"),
    avg_pmpm=("pmpm_cost","mean")
).reset_index()
if len(bench):
    bench["savings_pmpm"] = bench["avg_benchmark"] - bench["avg_pmpm"]
    figc = px.bar(bench, x="aco_id", y="savings_pmpm", title="Savings vs Benchmark (PMPM)")
    st.plotly_chart(figc, use_container_width=True)
    st.dataframe(bench)
else:
    st.info("No contracts for current filter.")

st.markdown("---")
st.caption("Prototype: de-identified synthetic data approximating real patterns; replace CSVs with your real datasets to operationalize.")