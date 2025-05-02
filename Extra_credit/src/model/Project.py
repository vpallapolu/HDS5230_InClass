#!/usr/bin/env python
# coding: utf-8

# In[781]:


pip install pandas 


# In[782]:


pip install scikit-learn 


# In[783]:


pip install seaborn 


# In[784]:


pip install matplotlib


# In[785]:


import pandas as pd
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


# In[786]:


get_ipython().system('curl -O https://raw.githubusercontent.com/vpallapolu/HDS5230_InClass/refs/heads/main/Extra_credit/data/health_activity_data.csv?token=GHSAT0AAAAAADC53J5Q6X6MGD7SL6PLWDEK2AVDTWQ')


# In[787]:


import pandas as pd
url = "health_activity_data.csv"
data = pd.read_csv(url)
data.head()


# In[788]:


data.describe()


# In[789]:


data.info()


# In[790]:


data.isnull().sum()


# gender encoding:
# 

# In[791]:


data = pd.get_dummies(data, columns=['Gender'], prefix='Gender', drop_first=False)


# In[792]:


data['Gender_Female'] = data['Gender_Female'].astype(int)
data['Gender_Male'] = data['Gender_Male'].astype(int)


# encode diabetic, heart disease, and smoker
# 

# In[793]:


for col in ['Smoker', 'Diabetic', 'Heart_Disease']:
    data[col] = LabelEncoder().fit_transform(data[col])


# In[794]:


data.head()


# split BP to numeric

# In[795]:


bp = data['Blood_Pressure'].str.split('/', expand=True)
data['SystolicBP'] = pd.to_numeric(bp[0], errors='coerce')
data['DiastolicBP'] = pd.to_numeric(bp[1], errors='coerce')


# drop  bp, id
# 

# In[796]:


data.drop(columns=['Blood_Pressure'], inplace=True)
data.drop(columns=['ID'], inplace=True)


# In[797]:


plt.figure(figsize=(10, 8))
sns.heatmap(data.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()


# In[798]:


data.head()


# **model fitting**

# In[799]:


data['Lifestyle_Score'] = (
    (data['Exercise_Hours_per_Week'] * 0.4) +    # Weight exercise hours
    (data['Calories_Intake'] * 0.3) +             # Weight calories intake
    (data['Hours_of_Sleep'] * 0.2) -              # Weight sleep hours
    (data['Smoker'] * 0.1) -                     # Deduct points for smoking
    (data['Alcohol_Consumption_per_Week'] * 0.1)   # Deduct points for alcohol consumption
)
data['Lifestyle_Score'].unique()
data['Lifestyle_Score'].describe()


# In[800]:


def classify_health_risk(row):
    if row['Lifestyle_Score'] < 528.03:
        return 'Low'
    elif row['Lifestyle_Score'] <= 866.6:
        return 'Medium'
    else:
        return 'High'

data['Health_Risk_score'] = data.apply(classify_health_risk, axis=1)
data.head()


# In[801]:


data.shape


# In[802]:


plt.figure(figsize=(6, 4))
sns.countplot(x=y, hue=y, palette='Set2', legend=False)  # Updated
plt.title("Distribution of Health Risk Score")
plt.xlabel("Health Risk Score (0=Low, 1=Medium, 2=High)")
plt.ylabel("Count")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


# In[803]:


# Ensure the target column is numeric
data['Health_Risk_score'] = data['Health_Risk_score'].map({'Low': 0, 'Medium': 1, 'High': 2})
# Check for any remaining non-numeric columns (besides the target and the columns to exclude)
# You might need to convert other columns such as Age, Height, Weight, etc. if they are still non-numeric
X = data.drop(columns=['Health_Risk_score', 'Lifestyle_Score'])  # Drop the target and outcome variables

# Ensure all features in X are numeric
X = pd.get_dummies(X, drop_first=True)  # This handles any remaining categorical columns (like if there are non-binary categories)

y = data['Health_Risk_score']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[804]:


# Standardize the features using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[805]:


# Initialize and train the Logistic Regression model
log_reg_model = LogisticRegression(max_iter=1000, multi_class='ovr', random_state=42)
log_reg_model.fit(X_train_scaled, y_train)


# In[806]:


# Predict the target variable on the test set
y_pred = log_reg_model.predict(X_test_scaled)


# In[807]:


# Evaluate the model performance
print("Logistic Regression Performance:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, average='weighted', zero_division=0))
print("Recall:", recall_score(y_test, y_pred, average='weighted', zero_division=0))
print("F1 Score:", f1_score(y_test, y_pred, average='weighted', zero_division=0))


# In[808]:


# Print the classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))


# In[809]:


# Confusion Matrix
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title("Logistic Regression Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


# In[810]:


X.columns


# In[ ]:




