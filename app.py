# app.py (Diabetes version)
import streamlit as st
import pandas as pd
import joblib, json, os
import plotly.express as px

st.set_page_config(page_title="Diabetes Predictor", layout="wide")

@st.cache_resource
def load_model(path="model.pkl"):
    if os.path.exists(path):
        return joblib.load(path)
    return None

@st.cache_data
def load_data(path="data/diabetes.csv"):
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

model = load_model("model.pkl")
df = load_data("data/diabetes.csv")

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home","Data","Visuals","Predict","Performance","About"])

if page == "Home":
    st.title("Diabetes Prediction App")
    st.markdown("Predict the presence of diabetes (Pima Indians dataset).")
    if df is not None:
        st.dataframe(df.head())
    else:
        st.warning("No dataset found at data/diabetes.csv")

elif page == "Data":
    st.header("Dataset Overview")
    if df is None:
        st.error("No dataset loaded.")
    else:
        st.write("Shape:", df.shape)
        st.dataframe(df.describe())
        st.subheader("Missing values")
        st.dataframe(df.isna().sum().to_frame("missing_count"))

elif page == "Visuals":
    st.header("Visualizations")
    if df is None:
        st.error("No dataset loaded.")
    else:
        st.subheader("Outcome distribution")
        fig = px.histogram(df, x='Outcome', title='Outcome distribution')
        st.plotly_chart(fig, use_container_width=True)

        if 'Age' in df.columns:
            fig2 = px.histogram(df, x='Age', nbins=20, title='Age distribution')
            st.plotly_chart(fig2, use_container_width=True)

        if 'Glucose' in df.columns and 'BMI' in df.columns:
            fig3 = px.scatter(df, x='Glucose', y='BMI', color='Outcome', title='Glucose vs BMI')
            st.plotly_chart(fig3, use_container_width=True)

elif page == "Predict":
    st.header("Make a prediction")
    if model is None:
        st.error("No model found (model.pkl). Upload and redeploy.")
    else:
        # Build inputs; default values based on typical ranges
        preg = st.number_input("Pregnancies", min_value=0, max_value=20, value=1)
        glucose = st.number_input("Glucose", min_value=0.0, max_value=300.0, value=120.0)
        bp = st.number_input("BloodPressure", min_value=0.0, max_value=150.0, value=70.0)
        skin = st.number_input("SkinThickness", min_value=0.0, max_value=100.0, value=20.0)
        insulin = st.number_input("Insulin", min_value=0.0, max_value=900.0, value=80.0)
        bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=32.0)
        dpf = st.number_input("DiabetesPedigreeFunction", min_value=0.0, max_value=5.0, value=0.5, format="%.3f")
        age = st.number_input("Age", min_value=1, max_value=120, value=30)

        if st.button("Predict"):
            input_df = pd.DataFrame([{
                "Pregnancies": preg, "Glucose": glucose, "BloodPressure": bp,
                "SkinThickness": skin, "Insulin": insulin, "BMI": bmi,
                "DiabetesPedigreeFunction": dpf, "Age": age
            }])
            try:
                pred = model.predict(input_df)[0]
                proba = model.predict_proba(input_df)[0][1] if hasattr(model, "predict_proba") else None
                st.success("Prediction: **Diabetes**" if pred==1 else "Prediction: **No diabetes**")
                if proba is not None:
                    st.info(f"Probability of diabetes: {proba:.2f}")
            except Exception as e:
                st.error("Prediction failed — check model compatibility and input columns.")
                st.exception(e)

elif page == "Performance":
    st.header("Model performance")
    if os.path.exists("artifacts/metrics.json"):
        with open("artifacts/metrics.json") as f:
            metrics = json.load(f)
        st.json(metrics)
    else:
        st.info("No artifacts/metrics.json found")

    cm_path = "artifacts/confusion_matrix.png"
    if os.path.exists(cm_path):
        st.image(cm_path, caption="Confusion matrix")
    else:
        st.info("No confusion matrix image found")

elif page == "About":
    st.header("About")
    st.markdown("""
    - Model pipeline saved as `model.pkl`
    - Dataset path: `data/diabetes.csv`
    - If model not uploaded, modify this app to download it at runtime or upload to repo.
    """)
