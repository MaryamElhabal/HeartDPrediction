import streamlit as st
import numpy as np 
import pandas as pd 
import pickle

with open('RainForestModel2.pkl', 'rb') as f:
    model = pickle.load(f)

st.title(":blue[ Heart Disease Prediction ] 🫀")
st.subheader("",divider="blue")
st.subheader(":gray[Enter patient data to predict the likelihood of heart disease ]")

# Define input fields
age = st.number_input(":blue[Age]", 18, 100, 45)
sex = st.selectbox(":blue[Sex]", ("Male", "Female"))
cp = st.selectbox(":blue[Chest Pain Type (cp)]", [0, 1, 2, 3])
trestbps = st.number_input(":blue[Resting Blood Pressure (mm Hg)]", 80, 200, 120)
chol = st.number_input(":blue[Serum Cholesterol (mg/dl)]", 100, 600, 240)
fbs = st.selectbox(":blue[Fasting Blood Sugar > 120 mg/dl (1 = True, 0 = False)]", [0, 1])
restecg = st.selectbox(":blue[Resting ECG Results]", [0, 1, 2])
thalach = st.number_input(":blue[Max Heart Rate Achieved]", 60, 250, 150)
exang = st.selectbox(":blue[Exercise Induced Angina (1 = Yes, 0 = No)]", [0, 1])
oldpeak = st.number_input(":blue[ST Depression Induced (Oldpeak)]", 0.0, 10.0, 1.0, step=0.1)
slope = st.selectbox(":blue[Slope of the Peak Exercise ST Segment]", [0, 1, 2])
ca = st.selectbox(":blue[Number of Major Vessels Colored by Fluoroscopy (0-4)]", [0, 1, 2, 3, 4])
thal = st.selectbox(":blue[Thalassemia (1 = Normal, 2 = Fixed Defect, 3 = Reversible Defect)]", [1, 2, 3])

input_data = np.array([[age, 1 if sex == "Male" else 0, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])

if st.button("Predict"):
    prediction = model.predict(input_data)

    st.header(":gray[Patient Data Summary]")
    columns = ['Age', 'Sex', 'Chest Pain Type', 'Resting BP', 'Cholesterol', 'FBS', 'Rest ECG',
           'Max HR', 'Exercise Angina', 'Oldpeak', 'Slope', 'CA', 'Thal']
    st.table(pd.DataFrame(input_data, columns=columns))

    normal_ranges = {
    "Indicator": [
        "Resting Blood Pressure",
        "Total Cholesterol",
        "Max Heart Rate",
        "Fasting Blood Sugar",
        "ST Depression (Oldpeak)",
        "Number of Major Vessels (CA)",
        "Thalassemia"
    ],
    "Normal Range": [
        "Less than 120 mmHg",
        "Less than 200 mg/dL",
        "100 - 170 bpm (depending on age)",
        "Less than 120 mg/dL",
        "Less than 2.0",
        "0",
        "1 (Normal)"
    ]
    }

    df = pd.DataFrame(normal_ranges)
    st.subheader("👩‍⚕️:blue[Normal Ranges for Heart Health] ")
    st.table(df)

    if prediction[0] == 1:
        st.error(f"🔴 High risk of heart disease.\nPlease consult a doctor for further medical evaluation.")
    else:
        st.success(f"🟢 Low risk of heart disease.\nContinue with healthy habits and regular checkups.")
