import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load scaler dan model
with open('model/scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

with open('model/lstm_model.pkl', 'rb') as file:
    lstm_model = pickle.load(file)
gncvhn
with open('model/svm_classifier.pkl', 'rb') as file:
    svm_classifier = pickle.load(file)

# Judul aplikasi
st.title('Prediksi Diabetes 🩺')

# Input fitur dari pengguna
st.header('Masukkan Nilai Fitur:')
Pregnancies = st.number_input('Pregnancies', min_value=0, max_value=20, value=1)
Glucose = st.number_input('Glucose', min_value=0, max_value=300, value=120)
BloodPressure = st.number_input('BloodPressure', min_value=0, max_value=200, value=80)
SkinThickness = st.number_input('SkinThickness', min_value=0, max_value=100, value=20)
Insulin = st.number_input('Insulin', min_value=0, max_value=900, value=80)
BMI = st.number_input('BMI', min_value=0.0, max_value=100.0, value=25.0)
DiabetesPedigreeFunction = st.number_input('DiabetesPedigreeFunction', min_value=0.0, max_value=2.5, value=0.5)
Age = st.number_input('Age', min_value=0, max_value=120, value=30)

# Gabungkan input menjadi DataFrame
input_features = pd.DataFrame([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]],
                               columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])

st.write("Input Data:", input_features)

def prediction(input_data):
    # Skala data menggunakan scaler
    scaled_data = scaler.transform(input_data)

    # Bentuk ulang input untuk LSTM
    input_data_lstm = scaled_data.reshape((scaled_data.shape[0], 1, scaled_data.shape[1]))

    # Ekstrak fitur menggunakan LSTM
    lstm_features = lstm_model.predict(input_data_lstm)

    # Prediksi menggunakan SVM
    prediction = svm_classifier.predict(lstm_features)

    if prediction:
        result = 'Positif Diabetes 🩺'
    else:
        result = 'Negatif Diabetes ✅'
    return result

# Tombol untuk melakukan prediksi
if st.button("Prediksi"):
    prediction_result = prediction(input_features)
    st.write("Hasil Prediksi: Pasien ini kemungkinan", prediction_result)
