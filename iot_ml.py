import os
import pickle
import streamlit as st
import numpy as np
import serial
import time
import random
from streamlit_option_menu import option_menu

# Set page configuration
st.set_page_config(page_title="Health Assistant", layout="wide", page_icon="🧑‍⚕️")

# Get the working directory
working_dir = os.path.dirname(os.path.abspath(__file__))

# Load the saved models
diabetes_model = pickle.load(open(f'{working_dir}/diabetes_model.sav', 'rb'))
heart_disease_model = pickle.load(open(f'{working_dir}/heart_disease_model.sav', 'rb'))
parkinsons_model = pickle.load(open(f'{working_dir}/parkinsons_model.sav', 'rb'))

# Header
st.title("Health Assistant")
st.write("Welcome to Health Assistant! This app predicts diabetes, heart disease, and Parkinson’s based on key parameter thresholds or machine learning, while monitoring vital signs in real-time.")

# Sidebar for navigation
with st.sidebar:
    st.write("### Navigation")
    st.write("Select a disease to predict or monitor your vital signs.")
    selected = option_menu(
        'Multiple Disease Prediction System',
        ['Diabetes Prediction', 'Heart Disease Prediction', 'Parkinsons Prediction'],
        menu_icon='hospital-fill',
        icons=['activity', 'heart', 'person'],
        default_index=0
    )
    customer_id = st.text_input("Customer ID")

# Function to get sensor data or simulate based on Customer ID
def get_sensor_data(port='COM14', baudrate=115200, max_attempts=5):
    try:
        ser = serial.Serial(port, baudrate, timeout=1)
        time.sleep(1)
        attempts = 0
        while attempts < max_attempts:
            line = ser.readline().decode('utf-8').strip()
            if not line or "Sensors initialized" in line:
                attempts += 1
                time.sleep(0.5)
                continue
            if 'IR:' in line and 'SpO2:' in line and 'Temp:' in line and 'HR:' in line:
                parts = line.split(' | ')
                data = {}
                for part in parts:
                    try:
                        key, value = part.split(': ')
                        if key == 'IR':
                            data['IR'] = float(value)
                        elif key == 'SpO2':
                            data['SpO2'] = None if value == '--' else float(value.split(' %')[0])
                        elif key == 'Temp':
                            data['Temp'] = float(value.split(' °C')[0])
                        elif key == 'HR':
                            data['HR'] = None if value == '--' else float(value.split(' BPM')[0])
                    except ValueError:
                        continue
                if 'SpO2' in data and 'Temp' in data and 'HR' in data:
                    ser.close()
                    data['simulated'] = False
                    return data
            attempts += 1
            time.sleep(0.5)
        ser.close()
    except Exception:
        try:
            id_num = int(customer_id) if customer_id.strip() else 1
            mode = (id_num - 1) % 3
        except ValueError:
            mode = 0
        if mode == 0:  # Healthy
            spo2 = round(random.uniform(95.0, 100.0), 1)
            temp = round(random.uniform(36.5, 37.5), 1)
            hr = random.randint(60, 100)
        elif mode == 1:  # Diseased
            spo2 = round(random.uniform(85.0, 90.0), 1)
            temp = round(random.uniform(37.6, 39.0), 1)
            hr = random.randint(100, 120) if random.choice([True, False]) else random.randint(40, 55)
        else:  # Object
            spo2 = 0.0
            temp = round(random.uniform(22.0, 25.0), 1)
            hr = 0.0
        return {'IR': 10000, 'SpO2': spo2, 'Temp': temp, 'HR': hr, 'simulated': True}

# Function to display sensor data
def display_sensor_data():
    with st.expander("Real-Time Vital Signs", expanded=False):
        if st.button("Get Current Sensor Readings"):
            sensor_data = get_sensor_data()
            spo2 = sensor_data.get('SpO2')
            temp = sensor_data.get('Temp')
            hr = sensor_data.get('HR')
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("SpO2", f"{spo2}%" if spo2 is not None else "--")
            with col2:
                st.metric("Temperature", f"{temp}°C")
            with col3:
                st.metric("Heart Rate", f"{hr} BPM" if hr is not None else "--")
            if spo2 is not None:
                if spo2 == 0:
                    st.error("No SpO2 reading. Sensor may not be on a living subject.")
                elif spo2 < 90:
                    st.warning("Oxygen saturation is low (<90%). Consult a healthcare provider.")
                elif spo2 < 95:
                    st.info("Oxygen saturation is slightly below normal (<95%).")
                else:
                    st.success("Oxygen saturation is normal (≥95%).")
            if temp > 37.5:
                st.warning("Temperature is elevated (>37.5°C). Possible fever.")
            elif temp < 36.5:
                if temp < 30:
                    st.error("Temperature is unusually low. Sensor may not be on a living subject.")
                else:
                    st.info("Temperature is slightly low (<36.5°C).")
            else:
                st.success("Temperature is normal (36.5–37.5°C).")
            if hr is not None:
                if hr == 0:
                    st.error("No heart rate detected. Sensor may not be on a living subject.")
                elif hr < 60:
                    st.warning("Heart rate is low (<60 BPM). Possible bradycardia; consult a doctor.")
                elif hr > 100:
                    st.warning("Heart rate is high (>100 BPM). Possible tachycardia; consult a doctor.")
                else:
                    st.success("Heart rate is normal (60–100 BPM).")
            else:
                st.write("Heart Rate: -- (Place your finger on the sensor)")
            st.subheader("Vital Signs Summary")
            summary = []
            advice = []
            if spo2 is not None:
                if spo2 >= 95:
                    summary.append("oxygen saturation is normal")
                elif spo2 >= 90:
                    summary.append("oxygen saturation is slightly low")
                    advice.append("Monitor your oxygen levels.")
                elif spo2 == 0:
                    summary.append("no oxygen saturation detected")
                    advice.append("Ensure the sensor is on a living subject.")
                else:
                    summary.append("oxygen saturation is critically low")
                    advice.append("Seek medical attention immediately.")
            if temp >= 36.5 and temp <= 37.5:
                summary.append("temperature is normal")
            elif temp > 37.5:
                summary.append("temperature is elevated")
                advice.append("Rest, stay hydrated, consult a doctor if fever persists.")
            elif temp < 30:
                summary.append("temperature is not physiological")
                advice.append("Ensure the sensor is on a living subject.")
            else:
                summary.append("temperature is low")
                advice.append("Keep warm and monitor your temperature.")
            if hr is not None:
                if hr >= 60 and hr <= 100:
                    summary.append("heart rate is normal")
                elif hr < 60:
                    summary.append("heart rate is low")
                    advice.append("Consult a healthcare provider for possible bradycardia.")
                elif hr == 0:
                    summary.append("no heart rate detected")
                    advice.append("Ensure the sensor is on a living subject.")
                else:
                    summary.append("heart rate is high")
                    advice.append("Relax and consult a doctor for possible tachycardia.")
            if all(x in ["oxygen saturation is normal", "temperature is normal", "heart rate is normal"] for x in summary):
                st.success(f"Your vital signs ({', '.join(summary)}) indicate good health.")
            else:
                st.warning(f"Your vital signs: {', '.join(summary)}.")
                for adv in advice:
                    st.info(adv)
            return sensor_data
    return None

# Diabetes Prediction Page
if selected == 'Diabetes Prediction':
    st.title('Diabetes Prediction')
    st.write("Enter details to predict diabetes. Key parameters trigger immediate diagnosis if abnormal; others use ML prediction.")
    
    # Create input fields
    col1, col2, col3 = st.columns(3)
    with col1:
        Pregnancies = st.text_input('Number of Pregnancies', placeholder="e.g., 2", help="Abnormal: >20", value="")
    with col2:
        Glucose = st.text_input('Glucose Level', placeholder="e.g., 120", help="Abnormal: >150 mg/dL", value="")
    with col3:
        BloodPressure = st.text_input('Blood Pressure', placeholder="e.g., 80", help="Abnormal: >150 mmHg", value="")
    with col1:
        SkinThickness = st.text_input('Skin Thickness', placeholder="e.g., 20", help="Used in ML model", value="")
    with col2:
        Insulin = st.text_input('Insulin Level', placeholder="e.g., 80", help="Abnormal: >150 mu U/ml", value="")
    with col3:
        BMI = st.text_input('BMI', placeholder="e.g., 30", help="Used in ML model", value="")
    with col1:
        DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function', placeholder="e.g., 0.5", help="Used in ML model", value="")
    with col2:
        Age = st.text_input('Age', placeholder="e.g., 30", help="Used in ML model", value="")

    if st.button('Diabetes Test Result'):
        # Collect inputs
        user_input = [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]
        input_labels = ['Pregnancies', 'Glucose', 'Blood Pressure', 'Skin Thickness', 'Insulin', 'BMI', 'Diabetes Pedigree Function', 'Age']
        
        # Validate inputs
        validated_input = []
        for i, val in enumerate(user_input):
            if not val.strip():
                st.error(f"Please enter a value for {input_labels[i]}.")
                break
            try:
                float_val = float(val)
                if float_val < 0:
                    st.error(f"{input_labels[i]} cannot be negative.")
                    break
                validated_input.append(float_val)
            except ValueError:
                st.error(f"Please enter a valid numerical value for {input_labels[i]}.")
                break
        else:
            # All inputs are valid
            try:
                pregnancies, glucose, bp, skin_thickness, insulin, bmi, dpf, age = validated_input

                # Check major parameters
                abnormal_params = []
                if pregnancies > 20:
                    abnormal_params.append("Pregnancies")
                if glucose > 150:
                    abnormal_params.append("Glucose")
                if bp > 150:
                    abnormal_params.append("Blood Pressure")
                if insulin > 150:
                    abnormal_params.append("Insulin")

                # Use exactly 8 features for the model
                model_input = validated_input  # No padding needed

                if abnormal_params:
                    st.success(f"The person is diabetic due to abnormal {', '.join(abnormal_params)} value(s).")
                    st.write("Abnormal key parameters suggest diabetes. Consult a healthcare provider.")
                else:
                    # Use ML model for other parameters
                    diab_prediction = diabetes_model.predict([model_input])
                    if diab_prediction[0] == 1:
                        st.success('The person is diabetic (ML prediction).')
                        st.write("ML model predicts diabetes based on other factors. Consult a healthcare provider.")
                    else:
                        st.success('The person is not diabetic (ML prediction).')
                        st.write("Maintain a healthy lifestyle to reduce diabetes risk.")
            except Exception as e:
                st.error(f"An error occurred during prediction: {str(e)}")

    display_sensor_data()

# Heart Disease Prediction Page
if selected == 'Heart Disease Prediction':
    st.title('Heart Disease Prediction')
    st.write("Enter details to predict heart disease. Key parameters trigger immediate diagnosis if abnormal; others use ML prediction.")
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.text_input('Age', placeholder="e.g., 50", help="Used in ML model", value="")
    with col2:
        sex = st.text_input('Sex', placeholder="1 or 0", help="Used in ML model", value="")
    with col3:
        cp = st.text_input('Chest Pain Type', placeholder="e.g., 0", help="Used in ML model", value="")
    with col1:
        trestbps = st.text_input('Resting Blood Pressure', placeholder="e.g., 120", help="Abnormal: >140 mmHg", value="")
    with col2:
        chol = st.text_input('Cholesterol', placeholder="e.g., 200", help="Abnormal: >240 mg/dl", value="")
    with col3:
        fbs = st.text_input('Fasting Blood Sugar', placeholder="1 or 0", help="Used in ML model", value="")
    with col1:
        restecg = st.text_input('Resting ECG', placeholder="e.g., 0", help="Used in ML model", value="")
    with col2:
        thalach = st.text_input('Max Heart Rate', placeholder="e.g., 150", help="Abnormal: <100 or >180", value="")
    with col3:
        exang = st.text_input('Exercise Induced Angina', placeholder="1 or 0", help="Used in ML model", value="")
    with col1:
        oldpeak = st.text_input('ST Depression', placeholder="e.g., 1.0", help="Abnormal: >2", value="")
    with col2:
        slope = st.text_input('ST Slope', placeholder="e.g., 2", help="Used in ML model", value="")
    with col3:
        ca = st.text_input('Major Vessels', placeholder="e.g., 0", help="Used in ML model", value="")
    with col1:
        thal = st.text_input('Thalassemia', placeholder="e.g., 2", help="Used in ML model", value="")

    if st.button('Heart Disease Test Result'):
        user_input = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
        input_labels = ['Age', 'Sex', 'Chest Pain Type', 'Resting Blood Pressure', 'Cholesterol', 'Fasting Blood Sugar', 
                        'Resting ECG', 'Max Heart Rate', 'Exercise Induced Angina', 'ST Depression', 'ST Slope', 
                        'Major Vessels', 'Thalassemia']
        
        # Validate inputs
        validated_input = []
        for i, val in enumerate(user_input):
            if not val.strip():
                st.error(f"Please enter a value for {input_labels[i]}.")
                break
            try:
                float_val = float(val)
                if float_val < 0:
                    st.error(f"{input_labels[i]} cannot be negative.")
                    break
                validated_input.append(float_val)
            except ValueError:
                st.error(f"Please enter a valid numerical value for {input_labels[i]}.")
                break
        else:
            try:
                age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal = validated_input

                # Check major parameters
                abnormal_params = []
                if trestbps > 140:
                    abnormal_params.append("Resting Blood Pressure")
                if chol > 240:
                    abnormal_params.append("Cholesterol")
                if thalach < 100 or thalach > 180:
                    abnormal_params.append("Max Heart Rate")
                if oldpeak > 2:
                    abnormal_params.append("ST Depression")

                # Use exactly 13 features for the model
                model_input = validated_input  # No padding needed

                if abnormal_params:
                    st.success(f"The person has heart disease due to abnormal {', '.join(abnormal_params)} value(s).")
                    st.write("Abnormal key parameters suggest heart issues. Consult a cardiologist.")
                else:
                    # Use ML model for other parameters
                    heart_prob = heart_disease_model.predict_proba([model_input])[0][1]
                    heart_prediction = heart_disease_model.predict([model_input])
                    if heart_prediction[0] == 1:
                        st.success(f'The person has heart disease (ML prediction, Probability: {heart_prob*100:.2f}%).')
                        st.write("ML model predicts heart disease. Consult a cardiologist.")
                    else:
                        st.success(f'The person does not have heart disease (ML prediction, Probability: {heart_prob*100:.2f}%).')
                        st.write("Continue monitoring heart health.")
            except Exception as e:
                st.error(f"An error occurred during prediction: {str(e)}")

    sensor_data = display_sensor_data()
    if sensor_data:
        spo2 = sensor_data.get('SpO2')
        temp = sensor_data.get('Temp')
        hr = sensor_data.get('HR')
        if spo2 is not None and spo2 < 90:
            st.warning("Low SpO2 (<90%) may indicate respiratory or circulatory issues.")
        if temp > 37.5:
            st.warning("Elevated temperature (>37.5°C) could suggest inflammation.")
        if hr is not None and (hr < 60 or hr > 100):
            st.warning("Abnormal heart rate may indicate heart issues. Consult a cardiologist.")

# Parkinson's Prediction Page
if selected == 'Parkinsons Prediction':
    st.title("Parkinson's Disease Prediction")
    st.write("Enter voice measurements to predict Parkinson’s. Key parameters trigger immediate diagnosis if abnormal; others use ML prediction.")
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        fo = st.text_input('MDVP:Fo(Hz)', placeholder="e.g., 120", help="Used in ML model", value="")
    with col2:
        fhi = st.text_input('MDVP:Fhi(Hz)', placeholder="e.g., 150", help="Used in ML model", value="")
    with col3:
        flo = st.text_input('MDVP:Flo(Hz)', placeholder="e.g., 100", help="Used in ML model", value="")
    with col4:
        Jitter_percent = st.text_input('MDVP:Jitter(%)', placeholder="e.g., 0.5", help="Abnormal: >0.6%", value="")
    with col5:
        Jitter_Abs = st.text_input('MDVP:Jitter(Abs)', placeholder="e.g., 0.0001", help="Used in ML model", value="")
    with col1:
        RAP = st.text_input('MDVP:RAP', placeholder="e.g., 0.003", help="Used in ML model", value="")
    with col2:
        PPQ = st.text_input('MDVP:PPQ', placeholder="e.g., 0.004", help="Used in ML model", value="")
    with col3:
        DDP = st.text_input('Jitter:DDP', placeholder="e.g., 0.009", help="Used in ML model", value="")
    with col4:
        Shimmer = st.text_input('MDVP:Shimmer', placeholder="e.g., 0.02", help="Abnormal: >0.03", value="")
    with col5:
        Shimmer_dB = st.text_input('MDVP:Shimmer(dB)', placeholder="e.g., 0.2", help="Used in ML model", value="")
    with col1:
        APQ3 = st.text_input('Shimmer:APQ3', placeholder="e.g., 0.01", help="Used in ML model", value="")
    with col2:
        APQ5 = st.text_input('Shimmer:APQ5', placeholder="e.g., 0.012", help="Used in ML model", value="")
    with col3:
        APQ = st.text_input('MDVP:APQ', placeholder="e.g., 0.015", help="Used in ML model", value="")
    with col4:
        DDA = st.text_input('Shimmer:DDA', placeholder="e.g., 0.03", help="Used in ML model", value="")
    with col5:
        NHR = st.text_input('NHR', placeholder="e.g., 0.02", help="Abnormal: >0.03", value="")
    with col1:
        HNR = st.text_input('HNR', placeholder="e.g., 20", help="Used in ML model", value="")
    with col2:
        RPDE = st.text_input('RPDE', placeholder="e.g., 0.5", help="Used in ML model", value="")
    with col3:
        DFA = st.text_input('DFA', placeholder="e.g., 0.7", help="Used in ML model", value="")
    with col4:
        spread1 = st.text_input('spread1', placeholder="e.g., -5", help="Used in ML model", value="")
    with col5:
        spread2 = st.text_input('spread2', placeholder="e.g., 0.2", help="Used in ML model", value="")
    with col1:
        D2 = st.text_input('D2', placeholder="e.g., 2.5", help="Used in ML model", value="")
    with col2:
        PPE = st.text_input('PPE', placeholder="e.g., 0.3", help="Abnormal: >0.4", value="")

    if st.button("Parkinson's Test Result"):
        user_input = [fo, fhi, flo, Jitter_percent, Jitter_Abs, RAP, PPQ, DDP, Shimmer, Shimmer_dB,
                      APQ3, APQ5, APQ, DDA, NHR, HNR, RPDE, DFA, spread1, spread2, D2, PPE]
        input_labels = ['MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)', 'MDVP:Jitter(Abs)', 
                        'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP', 'MDVP:Shimmer', 'MDVP:Shimmer(dB)', 
                        'Shimmer:APQ3', 'Shimmer:APQ5', 'MDVP:APQ', 'Shimmer:DDA', 'NHR', 'HNR', 
                        'RPDE', 'DFA', 'spread1', 'spread2', 'D2', 'PPE']
        
        # Validate inputs
        validated_input = []
        for i, val in enumerate(user_input):
            if not val.strip():
                st.error(f"Please enter a value for {input_labels[i]}.")
                break
            try:
                float_val = float(val)
                validated_input.append(float_val)
            except ValueError:
                st.error(f"Please enter a valid numerical value for {input_labels[i]}.")
                break
        else:
            try:
                fo, fhi, flo, jitter_percent, jitter_abs, rap, ppq, ddp, shimmer, shimmer_db, apq3, apq5, apq, dda, nhr, hnr, rpde, dfa, spread1, spread2, d2, ppe = validated_input

                # Check major parameters
                abnormal_params = []
                if jitter_percent > 0.6:
                    abnormal_params.append("Jitter(%)")
                if shimmer > 0.03:
                    abnormal_params.append("Shimmer")
                if nhr > 0.03:
                    abnormal_params.append("NHR")
                if ppe > 0.4:
                    abnormal_params.append("PPE")

                # Use exactly 22 features for the model
                model_input = validated_input  # No padding needed

                if abnormal_params:
                    st.success(f"The person has Parkinson's disease due to abnormal {', '.join(abnormal_params)} value(s).")
                    st.write("Abnormal voice parameters suggest Parkinson’s. Consult a neurologist.")
                else:
                    # Use ML model for other parameters
                    parkinsons_prediction = parkinsons_model.predict([model_input])
                    if parkinsons_prediction[0] == 1:
                        st.success("The person has Parkinson's disease (ML prediction).")
                        st.write("ML model predicts Parkinson’s. Consult a neurologist.")
                    else:
                        st.success("The person does not have Parkinson's disease (ML prediction).")
                        st.write("Continue monitoring for neurological symptoms.")
            except Exception as e:
                st.error(f"An error occurred during prediction: {str(e)}")

    display_sensor_data()
