#!/usr/bin/env python
# coding: utf-8

# In[167]:


import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import seaborn as sns
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier


# In[168]:


import urllib.request
url = 'https://raw.githubusercontent.com/vpallapolu/HDS5230_InClass/main/Extra_credit/data/health_activity_data.csv'
filename = 'health_activity_data.csv'
urllib.request.urlretrieve(url, filename)
data = pd.read_csv(filename)
data.head()


# In[169]:


data.describe()


# In[170]:


data.info()


# In[171]:


data.isnull().sum()


# Dropping unnecessary columns

# In[172]:


import numpy
data= data = data.drop(['BMI', 'ID'], axis=1)
data.head()


# categorizing male and female to 0 and 1

# In[173]:


data['Gender'].unique()


# In[174]:


data['Gender'] = data['Gender'].map({'Female': 0, 'Male': 1})


# In[175]:


data['Hypertension'] = data['Blood_Pressure'].apply(lambda bp: 1 if int(bp.split('/')[0]) > 120 or int(bp.split('/')[1]) > 80 else 0)
data = data.drop(['Blood_Pressure'], axis=1)
data.head()


# encode diabetic, heart disease, and smoker
# 

# In[176]:


for col in ['Smoker', 'Diabetic', 'Heart_Disease']:
    data[col] = LabelEncoder().fit_transform(data[col])
data.head()


# In[177]:


data.shape


# In[178]:


data.hist()


# In[179]:


plt.figure(figsize=(20, 8))
sns.heatmap(data.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()


# In[180]:


data.head()


# In[181]:


data.describe()


# In[182]:


def calculate_health_risk(row):
    score = 0
    # Daily Steps
    if row['Daily_Steps'] < 5000:
        score += 2
    elif row['Daily_Steps'] < 10000:
        score += 1
    # Calories
    if row['Calories_Intake'] > 2500:
        score += 2
    elif row['Calories_Intake'] > 2000:
        score += 1
    # Sleep
    if row['Hours_of_Sleep'] <= 7 or row['Hours_of_Sleep'] > 9:
        score += 2
    else:
        score += 0  # Sleep is within healthy range (>7 and <=9)
    # Heart Rate
    if row['Heart_Rate'] < 60 or row['Heart_Rate'] > 85:
        score += 2
    else:
        score += 0
    # Exercise
    if row['Exercise_Hours_per_Week'] < 3:
        score += 2
    elif row['Exercise_Hours_per_Week'] < 5:
        score += 1
    else:
        score += 0
    # Alcohol (Gender-specific)
    if row['Gender'] == 0:  # Female
        if row['Alcohol_Consumption_per_Week'] > 7:
            score += 2
        else:
            score += 0
    else:  # Male
        if row['Alcohol_Consumption_per_Week'] > 14:
            score += 2
        else:
            score += 0
    return score


# In[183]:


def classify_health_risk(score):
    if score <= 4:
        return 0  # Low
    elif score <= 6:
        return 1  # Medium
    else:
        return 2  # High


# In[184]:


data['Health_Risk'] = data.apply(calculate_health_risk, axis=1)
data['Health_Risk_Score'] = data['Health_Risk'].apply(classify_health_risk)


# In[185]:


data.describe()


# In[186]:


data.head()


# In[187]:


import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(6, 4))
sns.countplot(x='Health_Risk_Score', hue='Health_Risk_Score', data=data, palette='Set2', legend=False)
plt.title("Distribution of Health Risk Score")
plt.xlabel("Health Risk Score (0=Low, 1=Medium, 2=High)")
plt.ylabel("Count")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


# **model fitting**

# In[188]:


data.columns


# In[189]:


# Check for any remaining non-numeric columns (besides the target and the columns to exclude)
# You might need to convert other columns such as Age, Height, Weight, etc. if they are still non-numeric
X = data.drop(columns=['Health_Risk_Score', 'Health_Risk', 'Hypertension'])  # Drop the target and outcome variables
# Ensure all features in X are numeric
#X = pd.get_dummies(X, drop_first=True)  # This handles any remaining categorical columns (like if there are non-binary categories)
y = data['Health_Risk_Score']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[190]:


from xgboost import XGBClassifier
# Initialize the XGBoost classifier
xg_model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
# Train the model
xg_model.fit(X_train, y_train)


# In[191]:


# Predict the target variable on the test set
y_pred_xg = xg_model.predict(X_test)


# In[192]:


# Evaluate the model
print("XGBoost Performance:")
print("Accuracy:", accuracy_score(y_test, y_pred_xg))
print("Precision:", precision_score(y_test, y_pred_xg, average='weighted', zero_division=0))
print("Recall:", recall_score(y_test, y_pred_xg, average='weighted', zero_division=0))
print("F1 Score:", f1_score(y_test, y_pred_xg, average='weighted', zero_division=0))


# In[193]:


# Confusion Matrix
sns.heatmap(confusion_matrix(y_test, y_pred_xg), annot=True, fmt='d', cmap='Blues')
plt.title("XG Boost Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


# In[194]:


import joblib
joblib.dump(xg_model, "XG_Boost_model.pkl")


# In[195]:


X_train.columns


# In[ ]:




