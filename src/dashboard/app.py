"""Streamlit dashboard: interactive pricing + driver explanation."""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
from src.pipelines.predict import predict, explain

st.set_page_config(page_title="Insurance Pricing Intelligence", layout="wide")

st.title("Insurance Pricing & Risk Intelligence")
st.caption("Estimate annual claim cost and inspect the drivers behind each quote.")

tab1, tab2 = st.tabs(["Quote", "Dataset Insights"])

with tab1:
    c1, c2, c3 = st.columns(3)
    with c1:
        age = st.slider("Age", 18, 64, 35)
        sex = st.selectbox("Sex", ["male", "female"])
    with c2:
        bmi = st.slider("BMI", 15.0, 50.0, 28.0, 0.1)
        children = st.slider("Children", 0, 5, 1)
    with c3:
        smoker = st.selectbox("Smoker", ["no", "yes"])
        region = st.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])

    row = {"age": age, "sex": sex, "bmi": bmi, "children": children,
           "smoker": smoker, "region": region}

    if st.button("Generate quote", type="primary"):
        p = predict(row)
        exp = explain(row, top_k=8)

        m1, m2 = st.columns(2)
        m1.metric("Predicted annual cost", f"${p:,.0f}")
        m2.metric("Baseline (log-space)", f"{exp['base_value_log']:.3f}")

        df = pd.DataFrame(exp["top_contributions_log_space"])
        df["direction"] = df["shap_value"].apply(
            lambda v: "increases cost" if v > 0 else "decreases cost"
        )
        fig = px.bar(
            df, x="shap_value", y="feature", orientation="h",
            color="direction",
            color_discrete_map={"increases cost": "#d62728", "decreases cost": "#2ca02c"},
            title="Top drivers (SHAP, log-cost scale)",
        )
        fig.update_layout(yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True)


with tab2:
    proc = Path("data/processed/insurance_clean.csv")
    if not proc.exists():
        st.warning("Run `python -m src.features.build` first.")
    else:
        df = pd.read_csv(proc)
        c1, c2 = st.columns(2)
        c1.plotly_chart(px.histogram(df, x="charges", nbins=50,
                                     title="Charges distribution"),
                        use_container_width=True)
        c2.plotly_chart(px.box(df, x="smoker", y="charges", color="smoker",
                               title="Charges by smoker status"),
                        use_container_width=True)
        c3, c4 = st.columns(2)
        c3.plotly_chart(px.scatter(df, x="age", y="charges", color="smoker",
                                   title="Age vs charges"),
                        use_container_width=True)
        c4.plotly_chart(px.scatter(df, x="bmi", y="charges", color="smoker",
                                   title="BMI vs charges"),
                        use_container_width=True)
