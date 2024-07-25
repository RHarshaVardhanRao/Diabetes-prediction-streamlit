import streamlit as st
import pandas as pd
import numpy as np
import pickle
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# Load the pre-trained model and scaler
with open('best_model.pkl', 'rb') as f:
    best_model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Load the dataset
df = pd.read_csv("diabetes.csv")

# Streamlit app
st.title('Diabetes Prediction App')
st.write("This app uses machine learning to predict whether a person has diabetes based on their health information.")

# Sidebar inputs for new data
st.sidebar.header('Input Parameters')
def user_input_features():
    pregnancies = st.sidebar.slider('Pregnancies', 0, 20, 0)
    glucose = st.sidebar.slider('Glucose', 0, 200, 120)
    blood_pressure = st.sidebar.slider('Blood Pressure', 0, 140, 70)
    skin_thickness = st.sidebar.slider('Skin Thickness', 0, 99, 20)
    insulin = st.sidebar.slider('Insulin', 0, 846, 79)
    bmi = st.sidebar.slider('BMI', 0.0, 70.0, 32.0)
    diabetes_pedigree_function = st.sidebar.slider('Diabetes Pedigree Function', 0.0, 2.5, 0.5)
    age = st.sidebar.slider('Age', 0, 120, 33)
    
    data = {
        'Pregnancies': pregnancies,
        'Glucose': glucose,
        'BloodPressure': blood_pressure,
        'SkinThickness': skin_thickness,
        'Insulin': insulin,
        'BMI': bmi,
        'DiabetesPedigreeFunction': diabetes_pedigree_function,
        'Age': age
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Scale the input features
scaled_input = scaler.transform(input_df)

# Predict using the best model
if st.sidebar.button('Predict'):
    prediction = best_model.predict(scaled_input)
    st.subheader('Prediction')
    st.write('Diabetic' if prediction[0] == 1 else 'Non-diabetic')

# Space and line separator
st.write("---")

# Evaluation Metrics
X = df.drop('Outcome', axis=1)
y = df['Outcome']

X_scaled = scaler.transform(X)
y_pred_proba = best_model.predict_proba(X_scaled)[:, 1]

st.subheader('Confusion Matrix and Evaluation Metrics')

# Threshold slider
threshold = st.slider('Threshold', 0.0, 1.0, 0.5)

# Binarize predictions based on threshold
y_pred = (y_pred_proba >= threshold).astype(int)

# Confusion matrix
cm = confusion_matrix(y, y_pred)
st.subheader('Confusion Matrix')
st.write(f"Threshold: {threshold}")
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax)
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
ax.set_title('Confusion Matrix')
st.pyplot(fig)

# Evaluation metrics calculation
accuracy = accuracy_score(y, y_pred)
precision = precision_score(y, y_pred)
recall = recall_score(y, y_pred)
f1 = f1_score(y, y_pred)

st.subheader('Evaluation Metrics')
st.write(f'Accuracy: {accuracy:.2f}')
st.write(f'Precision: {precision:.2f}')
st.write(f'Recall: {recall:.2f}')
st.write(f'F1 Score: {f1:.2f}')

# AUC-ROC curve
fpr, tpr, thresholds = roc_curve(y, y_pred_proba)
roc_auc = roc_auc_score(y, y_pred_proba)

st.subheader('ROC Curve')
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(fpr, tpr, color='blue', label=f'ROC Curve (area = {roc_auc:.2f})')
ax.plot([0, 1], [0, 1], color='red', linestyle='--')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('Receiver Operating Characteristic (ROC) Curve')
ax.legend(loc="lower right")

# Mark the best threshold
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]
ax.scatter(fpr[optimal_idx], tpr[optimal_idx], marker='o', color='black', label=f'Best Threshold = {optimal_threshold:.2f}')
ax.legend(loc="lower right")
st.pyplot(fig)
