import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

# Page configuration
st.set_page_config(page_title="Diabetes Prediction", layout="centered")

# Custom CSS for styling
st.markdown("""
    <style>
        .main {
            background-color: #000000;
        }
        .sidebar .sidebar-content {
            background-color: #e9ecef;
        }
        .stButton>button {
            color: white;
            background: #28a745;
        }
        .stSlider>div>div>div>div {
            background: #28a745;
        }
        h1, h2, h3 {
            color: #28a745;
        }
        .css-1v0mbdj p, .css-1v0mbdj {
            color: #ffffff;
        }
        .css-1v0mbdj {
            color: #ffffff;
        }
        .css-1cpxqw2, .css-16huue1 {
            color: #ffffff;
        }
        .css-145kmo2 p {
            color: #ffffff;
        }
    </style>
""", unsafe_allow_html=True)

# Header
st.write("# Diabetes Prediction")

# Load data
@st.cache_data
def load_data():
    data = pd.read_csv("diabetes.csv")
    return data

data = load_data()

# Handling zero or less than zero values
cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for col in cols:
    zero_values = (data[col] <= 0).sum()
    if zero_values > 0:
        median = data[col].median()
        data.loc[data[col] <= 0, col] = median

# Outlier removal
def mod_outlier(df):
    df1 = df.copy()
    df = df._get_numeric_data()
    q1 = df.quantile(0.25)
    q3 = df.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - (1.5 * iqr)
    upper_bound = q3 + (1.5 * iqr)

    for col in df.columns:
        df1[col] = np.where(df[col] < lower_bound[col], lower_bound[col],
                            np.where(df[col] > upper_bound[col], upper_bound[col], df[col]))

    return df1

data_mod = mod_outlier(data)

# Splitting data
X = data_mod[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']]
y = data_mod['Outcome']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=43)

# Feature scaling
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Logistic Regression
clf = LogisticRegression(max_iter=100, multi_class="ovr", penalty="l1", solver="saga")
clf.fit(X_train_scaled, y_train)

# Function to calculate and display metrics and ROC curve
def display_metrics_and_roc_curve(threshold, feature_values):
    # Create a DataFrame with the feature values for prediction
    X_new = pd.DataFrame([feature_values], columns=X.columns)
    
    # Scale the feature values
    X_new_scaled = scaler.transform(X_new)
    
    # Predict probabilities
    probs = clf.predict_proba(X_test_scaled)[:, 1]
    y_pred = np.where(probs > threshold, 1, 0)
    
    # Predict the outcome for the new feature values
    new_prob = clf.predict_proba(X_new_scaled)[:, 1]
    new_pred = np.where(new_prob > threshold, 1, 0)[0]
    
    # AUC-ROC score
    roc_auc = roc_auc_score(y_test, probs)
    
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, probs)
    
    # Determine the best threshold
    J = tpr - fpr
    ix = np.argmax(J)
    best_threshold = thresholds[ix]

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    st.write("## Confusion Matrix:")
    st.write(pd.DataFrame(cm, columns=['Predicted 0', 'Predicted 1'], index=['Actual 0', 'Actual 1']))

    # Display evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    st.write("## Evaluation Metrics:")
    st.write(f"Accuracy: {accuracy:.2f}")
    st.write(f"Precision: {precision:.2f}")
    st.write(f"Recall: {recall:.2f}")
   
# Function to calculate and display metrics and ROC curve
def display_metrics_and_roc_curve(threshold, feature_values):
    # Create a DataFrame with the feature values for prediction
    X_new = pd.DataFrame([feature_values], columns=X.columns)
    
    # Scale the feature values
    X_new_scaled = scaler.transform(X_new)
    
    # Predict probabilities
    probs = clf.predict_proba(X_test_scaled)[:, 1]
    y_pred = np.where(probs > threshold, 1, 0)
    
    # Predict the outcome for the new feature values
    new_prob = clf.predict_proba(X_new_scaled)[:, 1]
    new_pred = np.where(new_prob > threshold, 1, 0)[0]
    
    # AUC-ROC score
    roc_auc = roc_auc_score(y_test, probs)
    
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, probs)
    
    # Determine the best threshold
    J = tpr - fpr
    ix = np.argmax(J)
    best_threshold = thresholds[ix]

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    st.write("## Confusion Matrix:")
    st.write(pd.DataFrame(cm, columns=['Predicted 0', 'Predicted 1'], index=['Actual 0', 'Actual 1']))

    # Display evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    st.write("## Evaluation Metrics:")
    st.write(f"Accuracy: {accuracy:.2f}")
    st.write(f"Precision: {precision:.2f}")
    st.write(f"Recall: {recall:.2f}")
    st.write(f"F1-score: {f1:.2f}")

    # Plot ROC curve
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label='ROC Curve (area = %0.2f)' % roc_auc)
    ax.plot([0, 1], [0, 1], linestyle='--')
    ax.scatter(fpr[ix], tpr[ix], marker='o', color='red', label='Best Threshold = %0.2f' % best_threshold)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic')
    ax.legend()
    st.write("## ROC Curve:")
    st.pyplot(fig)

    # Display AUC-ROC score
    st.write("## AUC-ROC Score:", roc_auc)

# Sidebar sliders for feature inputs
st.sidebar.write("## Adjust Feature Values")
feature_values = {
    'Pregnancies': st.sidebar.slider('Pregnancies', int(X['Pregnancies'].min()), int(X['Pregnancies'].max()), int(X['Pregnancies'].median())),
    'Glucose': st.sidebar.slider('Glucose', int(X['Glucose'].min()), int(X['Glucose'].max()), int(X['Glucose'].median())),
    'BloodPressure': st.sidebar.slider('BloodPressure', int(X['BloodPressure'].min()), int(X['BloodPressure'].max()), int(X['BloodPressure'].median())),
    'SkinThickness': st.sidebar.slider('SkinThickness', int(X['SkinThickness'].min()), int(X['SkinThickness'].max()), int(X['SkinThickness'].median())),
    'Insulin': st.sidebar.slider('Insulin', int(X['Insulin'].min()), int(X['Insulin'].max()), int(X['Insulin'].median())),
    'BMI': st.sidebar.slider('BMI', float(X['BMI'].min()), float(X['BMI'].max()), float(X['BMI'].median())),
    'DiabetesPedigreeFunction': st.sidebar.slider('DiabetesPedigreeFunction', float(X['DiabetesPedigreeFunction'].min()), float(X['DiabetesPedigreeFunction'].max()), float(X['DiabetesPedigreeFunction'].median())),
    'Age': st.sidebar.slider('Age', int(X['Age'].min()), int(X['Age'].max()), int(X['Age'].median()))
}

# Threshold slider
threshold = st.slider('Threshold:', 0.0, 1.0, 0.5)

# Display metrics and ROC curve based on the selected threshold and feature values
display_metrics_and_roc_curve(threshold, feature_values)

# Predict the outcome for the new feature values
X_new = pd.DataFrame([feature_values], columns=X.columns)
X_new_scaled = scaler.transform(X_new)
predicted_probability = clf.predict_proba(X_new_scaled)[0][1]
predicted_outcome = clf.predict(X_new_scaled)[0]

# Display predicted values
st.write("## Prediction for Input Feature Values:")
st.write(f"Predicted Probability: {predicted_probability:.2f}")
st.write(f"Predicted Outcome: {predicted_outcome}")
