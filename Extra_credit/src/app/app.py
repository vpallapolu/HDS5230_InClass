#!/usr/bin/env python
# coding: utf-8

# In[54]:


import joblib
import numpy as np
import streamlit as st
import pandas as pd


# In[55]:


st.title("Heart Disease Risk Predictor (Random Forest)")
st.markdown("Enter your details to check your heart disease risk.")


# In[56]:


model = joblib.load("XG_Boost_model.pkl")


# In[57]:


# Input fields
age = st.number_input("Age", 1, 120, 30)
gender = st.selectbox("Gender", ["Male", "Female"])
height = st.number_input("Height (cm)", 100, 250, 170)
weight = st.number_input("Weight (kg)", 30, 200, 70)
daily_steps = st.number_input("Daily Steps", 0, 20000, 5000)
calories_intake = st.number_input("Calories Intake (kcal)", 0, 5000, 2000)
hours_of_sleep = st.number_input("Hours of Sleep", 0, 24, 7)
alcohol_consumption = st.number_input("Alcohol Consumption (drinks/week)", 0, 100, 0)
smoker = st.selectbox("Smoker", ["Yes", "No"])
diabetic = st.selectbox("Diabetic", ["Yes", "No"])
heart_rate = st.number_input("Heart Rate", 40, 180, 75)
exercise = st.number_input("Exercise Hours/Week", 0, 20, 3)
heart_Disease = st.selectbox("Heart Disease", ["Yes", "No"])


# In[58]:


#Button to make prediction 
#This button will trigger the prediction when clicked
st.markdown("Enter your details and click the button to predict your heart disease risk.")
if st.button("Predict"): # Need to make sure the input is in the right format and right order for the model to predict
    user_input = pd.DataFrame([{
        'Age': age,
        'Gender': 1 if gender == "Male" else 0,
        'Height_cm': height,
        'Weight_kg': weight,
        'Daily_Steps': daily_steps,
        'Calories_Intake': calories_intake,
        'Hours_of_Sleep': hours_of_sleep,
        'Heart_Rate': heart_rate,
        'Exercise_Hours_per_Week': exercise,
        'Smoker': 1 if smoker == "Yes" else 0,
        'Alcohol_Consumption_per_Week': alcohol_consumption,
        'Diabetic': 1 if diabetic == "Yes" else 0,
        'Heart_Disease': 1 if heart_Disease == "Yes" else 0}])
    prediction = model.predict(user_input)
    # Map numeric prediction to label
    risk_labels = {0: "Low", 1: "Medium", 2: "High"}
    risk_level = risk_labels.get(prediction[0], "Unknown")
    st.success(f"Predicted Health Risk Score: **{risk_level} Risk**")

