"""
Crop Recommendation System - Web App (demo)
--------------------------------------------
Run:  streamlit run app.py

Enter soil and weather conditions, and the trained Random Forest model
recommends the most suitable crop.
"""

import os
import joblib
import pandas as pd
import streamlit as st

DATA_PATH = os.path.join("data", "Crop_recommendation.csv")
MODEL_PATH = os.path.join("outputs", "crop_rf_model.pkl")
FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]

st.set_page_config(page_title="Crop Recommendation", page_icon="🌾", layout="wide")


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error("Model not found. Run `python model.py` first to train and save the model.")
        st.stop()
    return joblib.load(MODEL_PATH)


df = load_data()
model = load_model()
crop_means = df.groupby("label")[FEATURES].mean()

# Example inputs for the demo (typical conditions taken from the dataset)
EXAMPLES = {
    "Custom": None,
    "Rice-like conditions": dict(N=80, P=48, K=40, temperature=23.7, humidity=82.3, ph=6.4, rainfall=236.0),
    "Cotton-like conditions": dict(N=118, P=46, K=20, temperature=24.0, humidity=79.8, ph=6.9, rainfall=80.0),
    "Apple-like conditions": dict(N=21, P=134, K=200, temperature=22.6, humidity=92.3, ph=5.9, rainfall=113.0),
    "Coffee-like conditions": dict(N=101, P=29, K=30, temperature=25.5, humidity=58.9, ph=6.8, rainfall=158.0),
    "Chickpea-like conditions": dict(N=40, P=68, K=80, temperature=18.9, humidity=16.9, ph=7.3, rainfall=80.0),
}

# (label, column, min, max, default, step, unit)
INPUTS = [
    ("Nitrogen (N)", "N", 0, 140, 50, 1, "kg/ha"),
    ("Phosphorus (P)", "P", 5, 145, 50, 1, "kg/ha"),
    ("Potassium (K)", "K", 5, 205, 50, 1, "kg/ha"),
    ("Temperature", "temperature", 8.0, 44.0, 25.0, 0.1, "°C"),
    ("Humidity", "humidity", 14.0, 100.0, 70.0, 0.1, "%"),
    ("Soil pH", "ph", 3.5, 10.0, 6.5, 0.1, ""),
    ("Rainfall", "rainfall", 20.0, 300.0, 100.0, 1.0, "mm"),
]

# ---------------- Sidebar: inputs ----------------
st.sidebar.header("🌱 Enter Field Conditions")

for _, col, lo, hi, default, step, unit in INPUTS:
    st.session_state.setdefault(col, default)

example = st.sidebar.selectbox("Load an example", list(EXAMPLES.keys()))
if example != "Custom" and st.session_state.get("last_example") != example:
    for _, col, lo, hi, default, step, unit in INPUTS:
        st.session_state[col] = type(default)(EXAMPLES[example][col])
st.session_state["last_example"] = example

values = {}
for i, (label, col, lo, hi, default, step, unit) in enumerate(INPUTS):
    if i == 0:
        st.sidebar.subheader("Soil nutrients")
    elif i == 3:
        st.sidebar.subheader("Weather & soil")
    name = f"{label} [{unit}]" if unit else label
    values[col] = st.sidebar.slider(name, lo, hi, step=step, key=col)

# ---------------- Main page ----------------
st.title("🌾 Crop Recommendation System")
st.caption("Precision agriculture using Random Forest · FAI Case Study · "
           "Maddula Karthik (CH.SC.U4CSE23121), Amrita Vishwa Vidyapeetham, Chennai")

tab_predict, tab_model, tab_data = st.tabs(["🔍 Recommend", "📊 Model Performance", "📁 Dataset"])

with tab_predict:
    st.subheader("Your input")
    st.dataframe(pd.DataFrame([values]), hide_index=True, width="stretch")

    if st.button("Recommend Crop", type="primary", width="stretch"):
        sample = pd.DataFrame([values], columns=FEATURES)
        probs = pd.Series(model.predict_proba(sample)[0], index=model.classes_)
        top3 = probs.sort_values(ascending=False).head(3)
        best = top3.index[0]

        st.success(f"## ✅ Recommended crop: **{best.upper()}**")
        st.metric("Model confidence", f"{top3.iloc[0]:.0%}",
                  help="Share of the 100 decision trees that voted for this crop")

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Top 3 suggestions")
            top3_df = pd.DataFrame({"Crop": top3.index.str.title(), "Confidence": top3.values})
            st.bar_chart(top3_df, x="Crop", y="Confidence", horizontal=True)
        with c2:
            st.markdown(f"#### Your field vs. ideal for {best.title()}")
            compare = pd.DataFrame({
                "Your input": [values[f] for f in FEATURES],
                f"Average for {best}": crop_means.loc[best].round(1).values,
            }, index=FEATURES)
            st.dataframe(compare, width="stretch")
    else:
        st.info("Adjust the sliders on the left (or load an example), then click **Recommend Crop**.")

with tab_model:
    st.subheader("How well does the model work?")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Test accuracy", "99.55%")
    m2.metric("5-fold CV accuracy", "99.55%")
    m3.metric("Crops supported", "22")
    m4.metric("Trees in forest", "100")
    charts = [
        ("07_feature_importance.png", "Which conditions matter most"),
        ("06_confusion_matrix.png", "Confusion matrix on 440 test samples"),
        ("09_model_comparison.png", "Comparison with other algorithms"),
        ("08_accuracy_vs_trees.png", "Accuracy vs number of trees"),
    ]
    for i in range(0, len(charts), 2):
        cols = st.columns(2)
        for col, (fname, caption) in zip(cols, charts[i:i + 2]):
            path = os.path.join("outputs", fname)
            if os.path.exists(path):
                col.image(path, caption=caption, width="stretch")

with tab_data:
    st.subheader("Crop Recommendation Dataset (Kaggle)")
    st.write(f"{len(df)} samples · {len(FEATURES)} features · {df['label'].nunique()} crops "
             "(100 samples each)")
    st.dataframe(df, width="stretch", height=300)
    st.markdown("#### Average conditions for each crop")
    st.dataframe(crop_means.round(1), width="stretch")
