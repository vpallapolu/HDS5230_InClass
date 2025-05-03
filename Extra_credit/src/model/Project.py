#!/usr/bin/env python
# coding: utf-8

# In[1431]:


import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier


# In[1432]:


import urllib.request

url = 'https://raw.githubusercontent.com/vpallapolu/HDS5230_InClass/main/Extra_credit/data/health_activity_data.csv'
filename = 'health_activity_data.csv'
urllib.request.urlretrieve(url, filename)
data = pd.read_csv(filename)
data.head()


# In[1433]:


data.describe()


# In[1434]:


data.info()


# In[1435]:


data.isnull().sum()


# Dropping unnecessary columns

# In[1436]:


import numpy
data= data = data.drop(['BMI', 'ID'], axis=1)
data.head()


# categorizing male and female to 0 and 1

# In[1437]:


data['Gender'].unique()


# In[1438]:


data['Gender'] = data['Gender'].map({'Female': 0, 'Male': 1})


# In[1439]:


data['Hypertension'] = data['Blood_Pressure'].apply(lambda bp: 1 if int(bp.split('/')[0]) > 120 or int(bp.split('/')[1]) > 80 else 0)
data = data.drop(['Blood_Pressure'], axis=1)
data.head()


# encode diabetic, heart disease, and smoker
# 

# In[1440]:


for col in ['Smoker', 'Diabetic', 'Heart_Disease']:
    data[col] = LabelEncoder().fit_transform(data[col])
data.head()


# In[1441]:


data.shape


# In[1442]:


data.hist()


# In[1443]:


top_corr = high_corr_filtered.head(10) 
top_corr


# In[1444]:


plt.figure(figsize=(20, 8))
sns.heatmap(data.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()


# In[1445]:


data.head()


# In[1446]:


data.describe()


# In[1447]:


import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(6, 4))
sns.countplot(x='Heart_Disease', hue='Heart_Disease', data=data, palette='Set2', legend=False)
plt.title("Distribution of Heart Disease")
plt.xlabel("Heart Disease (0=Low, 1=High)")
plt.ylabel("Count")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


# **model fitting**

# In[1448]:


data.columns


# In[1449]:


# Check for any remaining non-numeric columns (besides the target and the columns to exclude)
# You might need to convert other columns such as Age, Height, Weight, etc. if they are still non-numeric
X = data.drop(columns=['Daily_Steps', 'Calories_Intake', 'Hours_of_Sleep', 'Alcohol_Consumption_per_Week' , 'Heart_Disease'])  # Drop the target and outcome variables
# Ensure all features in X are numeric
#X = pd.get_dummies(X, drop_first=True)  # This handles any remaining categorical columns (like if there are non-binary categories)
y = data['Heart_Disease']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[1450]:


# Initialize and train the Random Forest model
# Train model
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)


# In[1451]:


# Predict the target variable on the test set
y_pred_rf = rf_model.predict(X_test)


# In[1452]:


# Evaluate the model performance
print("Random Forest Performance:")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Precision:", precision_score(y_test, y_pred_rf, average='weighted', zero_division=0))
print("Recall:", recall_score(y_test, y_pred_rf, average='weighted', zero_division=0))
print("F1 Score:", f1_score(y_test, y_pred_rf, average='weighted', zero_division=0))


# In[1453]:


# Prepare predictors and target
X = data.drop(columns=['Daily_Steps', 'Calories_Intake', 'Hours_of_Sleep',
                       'Alcohol_Consumption_per_Week', 'Heart_Disease'])

X = pd.get_dummies(X, drop_first=True)
y = data['Heart_Disease']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[1454]:


# Print the classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred_rf, zero_division=0))


# In[1455]:


# Confusion Matrix
sns.heatmap(confusion_matrix(y_test, y_pred_rf), annot=True, fmt='d', cmap='Blues')
plt.title("Random Forest Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


# In[1456]:


import joblib
joblib.dump(rf_model, "Random_Forest_model.pkl")
joblib.dump(X.columns.tolist(), 'Random_Forest_model_features.pkl')


# In[1457]:


import streamlit as st
st.title("Heart Disease Risk Predictor (Random Forest)")
st.markdown("Enter your details to check your heart disease risk.")
# User input
age = st.number_input("Age", 1, 120, 30)
gender = st.selectbox("Gender", ["Male", "Female"])
height = st.number_input("Height (cm)", 100, 250, 170)
weight = st.number_input("Weight (kg)", 30, 200, 70)
hypertension = st.selectbox("Hypertension", ["Yes", "No"])
smoker = st.selectbox("Smoker", ["Yes", "No"])
diabetic = st.selectbox("Diabetic", ["Yes", "No"])
heart_rate = st.number_input("Heart Rate", 40, 180, 75)
exercise = st.number_input("Exercise Hours/Week", 0, 20, 3)
# Input dictionary
input_dict = {
    'Age': age,
    'Height_cm': height,
    'Weight_kg': weight,
    'Heart_Rate': heart_rate,
    'Exercise_Hours_per_Week': exercise,
    'Gender_Male': 1 if gender == "Male" else 0,
    'Smoker_Yes': 1 if smoker == "Yes" else 0,
    'Diabetic_Yes': 1 if diabetic == "Yes" else 0,
    'Hypertension_Yes': 1 if hypertension == "Yes" else 0}
# Convert to DataFrame and add missing columns
input_df = pd.DataFrame([input_dict])
for col in features:
    if col not in input_df.columns:
        input_df[col] = 0
input_df = input_df[features]
# Prediction
if st.button("Predict"):
    prediction = model.predict(input_df)[0]
    if prediction == 1:
        st.error("⚠️ High risk of heart disease!")
    else:
        st.success("✅ Low risk of heart disease.")


# In[ ]:




