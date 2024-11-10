import streamlit as st
import pandas as pd 
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from gradio_client import Client
# Add custom CSS for better styling
st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
    }
    h1 {
        color: #2c3e50;
        text-align: center;
        font-family: 'Arial', sans-serif;
    }
    .stButton > button {
        background-color: #2ecc71;
        color: white;
        font-size: 16px;
        padding: 10px;
    }
    .stRadio, .stNumberInput {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1);
    }
    .severity-box {
        padding: 10px;
        border-radius: 5px;
        font-size: 18px;
        color: white;
        text-align: center;
        margin-top: 10px;
    }
    .low-risk {
        background-color: #27ae60;
    }
    .moderate-risk {
        background-color: #f39c12;
    }
    .high-risk {
        background-color: #e74c3c;
    }
    </style>
    """, unsafe_allow_html=True)

# Load and preprocess the dataset
lung_cancer_df = pd.read_csv("lung_cancer_exported.csv")

temp_df = lung_cancer_df[['AGE', 'SMOKING', 'YELLOW_FINGERS', 'FATIGUE ', 'ALLERGY ', 'COUGHING',
              'CHRONIC DISEASE', 'SWALLOWING DIFFICULTY', 'CHEST PAIN', 'LUNG_CANCER']]

label_encoder = LabelEncoder()
for i in ['AGE','SMOKING', 'YELLOW_FINGERS', 'FATIGUE ', 'ALLERGY ', 'COUGHING',
          'CHRONIC DISEASE', 'SWALLOWING DIFFICULTY', 'CHEST PAIN', 'LUNG_CANCER']:
    temp_df[i] = label_encoder.fit_transform(temp_df[i])

X = temp_df.drop(columns=['LUNG_CANCER'])
y = temp_df['LUNG_CANCER']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

def get_severity(prob):
    if prob < 0.4:
        return "Low Risk", "low-risk"
    elif prob < 0.7:
        return "Moderate Risk", "moderate-risk"
    else:
        return "High Risk", "high-risk"

st.title("🌿 Lung Cancer Prediction Dashboard")

st.write("""
    Welcome to the **Lung Cancer Prediction** tool. 
    Fill out the following details to get a prediction and risk analysis based on your input.
    """)

# Create two columns for input fields
col1, col2 = st.columns(2)

# Column 1 inputs
with col1:
    age = st.number_input("Enter Age:", min_value=0, max_value=120, step=1, help="Your current age.")
    smoking = st.radio("Do you smoke?", ['Yes', 'No'])
    yellow = st.radio("Do you have yellow fingers?", ['Yes', 'No'])
    fatigue = st.radio("Do you feel tired all the time?", ['Yes', 'No'])
    allergy = st.radio("Are you allergic to anything?", ['Yes', 'No'])

# Column 2 inputs
with col2:
    cough = st.radio("Is your coughing more than usual?", ['Yes', 'No'])
    chronic = st.radio("Do you have any chronic disease?", ['Yes', 'No'])
    swallowing = st.radio("Do you have difficulty swallowing?", ['Yes', 'No'])
    chest = st.radio("Do you have frequent chest pain?", ['Yes', 'No'])

# Convert 'Yes' to 1 and 'No' to 0
smoking = 1 if smoking == 'Yes' else 0
yellow = 1 if yellow == 'Yes' else 0
fatigue = 1 if fatigue == 'Yes' else 0
allergy = 1 if allergy == 'Yes' else 0
cough = 1 if cough == 'Yes' else 0
chronic = 1 if chronic == 'Yes' else 0
swallowing = 1 if swallowing == 'Yes' else 0
chest = 1 if chest == 'Yes' else 0

if st.button("Predict"):
    input_data = pd.DataFrame([[age, smoking, yellow, fatigue, allergy, cough, chronic , swallowing, chest]], columns=X_train.columns)
    pred = clf.predict(input_data)
    prob = clf.predict_proba(input_data)[:, 1][0]

    severity, css_class = get_severity(prob)
    
    st.write(f"### Prediction: {'Lung Cancer' if pred == 1 else 'No Lung Cancer'}")
    st.write(f"#### Probability: {prob:.2f}")
    
    # Display severity with color
    st.markdown(f'<div class="severity-box {css_class}">Risk Level: {severity}</div>', unsafe_allow_html=True)
    
    result = ""
    s = f"""You are an AI medical advisor. Based on the following patient details, please provide a recommendation regarding potential next steps, including further screening, lifestyle modifications, or medical consultations. The patient details are as follows:
    Age: {age}
    Smoking status: {smoking} (1 = Yes, 0 = No)
    Yellow Fingers: {yellow} (1 = Yes, 0 = No)
    Fatigue: {fatigue} (1 = Yes, 0 = No)
    Allergy: {allergy} (1 = Yes, 0 = No)
    Coughing: {cough} (1 = Yes, 0 = No)
    Chronic Disease: {chronic} (1 = Yes, 0 = No)
    Swallowing Difficulty: {swallowing} (1 = Yes, 0 = No)
    Chest Pain: {chest} (1 = Yes, 0 = No)
    The model predicts the probability of lung cancer for this patient is {prob}, with a severity rating of {severity} (Low, Moderate, or High Risk).

    Based on this information, please provide the following recommendations:

    Immediate actions: Any medical consultations, diagnostic tests, or screenings the patient should consider based on their risk level.
    Lifestyle adjustments: Specific steps the patient can take to reduce their lung cancer risk or mitigate symptoms.
    Long-term considerations: Any strategies or monitoring that may help this patient reduce future risks or stay on top of their health.
    Please ensure the recommendations are tailored to the risk level (Low, Moderate, or High Risk) and patient details."""
    
    from huggingface_hub import InferenceClient

    client = InferenceClient(api_key="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")

    messages = [
        { "role": "user", "content": s }
    ]

    stream = client.chat.completions.create(
        model="HuggingFaceH4/zephyr-7b-beta", 
        messages=messages, 
        temperature=0.5,
        max_tokens=1024,
        top_p=0.7,
        stream=True
    )
    
    for chunk in stream:
        result += chunk.choices[0].delta.content
    st.write("\n")
    st.write("\n")
    st.write(f"### Recommendations from our side")
    st.write(result)