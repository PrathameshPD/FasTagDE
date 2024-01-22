#!/usr/bin/env python
# coding: utf-8

# In[1]:


from distutils.sysconfig import get_python_inc
from sysconfig import get_python_version


get_python_inc().system('pip install shap')


# In[2]:


get_python_version().system('pip install ipywidgets')


# In[3]:


#Data Manipulation-----------
import pandas as pd
import numpy as np

#Data Visualization----------
import matplotlib.pyplot as plt
import seaborn as sns
import phik
from phik.report import plot_correlation_matrix

#Stats-----------------------
import statsmodels.api as sm
from scipy import stats

#Data Preprocessing-----------
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

#Model-----------------------
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix

#Metrics----------------------
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score 

#Shap-------------------------
import shap


# In[4]:


data = pd.read_csv("C:\\Users\\Panda\\Desktop\\FinalFTG\\FastagFraudDetection.csv")


# In[5]:


data.head()


# In[6]:


data.tail()


# In[7]:


data.info()


# In[8]:


data.describe()


# In[9]:


data.shape


# In[10]:


data.size


# In[11]:


data.columns


# In[12]:


data.isnull().sum()


# In[13]:


FTG = 'FastagID'
# 1. Fill missing values with a specific value (e.g., 0)
data[FTG] = data[FTG].fillna(0)


# In[14]:


data.isnull().sum()


# In[15]:


data.nunique()


# In[16]:


#Converting 'Timestamp' to datetime format
data['Timestamp'] = pd.to_datetime(data['Timestamp'])

# Encoding categorical variables (initially with label encoding for simplicity)
from sklearn.preprocessing import LabelEncoder

# Creating a copy of the dataset for encoding
encoded_data = data.copy()

# List of categorical columns to encode
categorical_columns = ['Vehicle_Type', 'FastagID', 'TollBoothID', 'Lane_Type', 
                       'Vehicle_Dimensions', 'Geographical_Location', 'Vehicle_Plate_Number', 'Fraud_indicator']

# Applying label encoding
label_encoder = LabelEncoder()
for col in categorical_columns:
    encoded_data[col] = label_encoder.fit_transform(encoded_data[col].astype(str))

# Outlier detection and handling (simple method using IQR for this plan)
numerical_columns = ['Transaction_Amount', 'Amount_paid', 'Vehicle_Speed']

# Calculating IQR for each numerical column
Q1 = encoded_data[numerical_columns].quantile(0.25)
Q3 = encoded_data[numerical_columns].quantile(0.75)
IQR = Q3 - Q1

# Filtering out the outliers by keeping only the valid values
filtered_data = encoded_data[~((encoded_data[numerical_columns] < (Q1 - 1.5 * IQR)) | (encoded_data[numerical_columns] > (Q3 + 1.5 * IQR))).any(axis=1)]

# Displaying missing values and a sample of the cleaned data
filtered_data.head()


# In[17]:


import matplotlib.pyplot as plt
import seaborn as sns

# Setting plot style
plt.style.use('ggplot')

# Statistical Summary of the Numerical Columns
numerical_summary = filtered_data.describe()

# Distribution Analysis for Key Numerical Variables
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Histograms for Transaction_Amount, Amount_paid, and Vehicle_Speed
sns.histplot(filtered_data['Transaction_Amount'], bins=30, ax=axes[0], kde=True)
axes[0].set_title('Transaction Amount Distribution')

sns.histplot(filtered_data['Amount_paid'], bins=30, ax=axes[1], kde=True)
axes[1].set_title('Amount Paid Distribution')

sns.histplot(filtered_data['Vehicle_Speed'], bins=30, ax=axes[2], kde=True)
axes[2].set_title('Vehicle Speed Distribution')

plt.tight_layout()

# Correlation Analysis
correlation_matrix = filtered_data.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()



# In[18]:


# Count plot for Vehicle Type
plt.figure(figsize=(10, 6))
sns.countplot(x='Vehicle_Type', data=data)
plt.title('Count of Transactions by Vehicle Type')
plt.xlabel('Vehicle Type')
plt.ylabel('Transaction Count')
plt.show()


# In[19]:


#Countplot for Lane Type
plt.figure(figsize=(8, 5))
sns.countplot(x='Lane_Type', data=data, palette='pastel')
plt.title('Distribution of Lane Types')
plt.xlabel('Lane Type')
plt.ylabel('Count')
plt.show()


# In[20]:


# Lane usage comparison
plt.figure(figsize=(10, 6))
data.groupby("Lane_Type").agg(count=("Transaction_ID", "count"), mean_amount=("Transaction_Amount", "mean")).plot(kind="bar")
plt.title("Lane Usage Comparison")
plt.xlabel("Lane Type")
plt.ylabel("Count (left axis), Average Amount (right axis)")
plt.show()

# Boxplot of transaction amount by lane type
plt.figure(figsize=(10, 6))
sns.boxplot(x="Lane_Type", y="Transaction_Amount", showmeans=True, data=data
           )
plt.title("Distribution of Transaction Amount by Lane Type")
plt.show()


# In[21]:


# Boxplot of transaction amount by fraud indicator
plt.figure(figsize=(10, 6))
sns.boxplot(x="Fraud_indicator", y="Transaction_Amount", showmeans=True, data=data)
plt.title("Distribution of Transaction Amount by Fraud Indicator")
plt.show()


# In[22]:


data['is_duplicate_fastag'] = data.duplicated(subset='FastagID', keep=False)
data['is_duplicate_plate'] = data.duplicated(subset='Vehicle_Plate_Number', keep=False)


# In[23]:


label_encoder = LabelEncoder()
data['Vehicle_Type'] = label_encoder.fit_transform(data['Vehicle_Type'])
data['Lane_Type'] = label_encoder.fit_transform(data['Lane_Type'])
data['Geographical_Location'] = label_encoder.fit_transform(data['Geographical_Location'])

for col in ['Transaction_Amount', 'Amount_paid', 'Vehicle_Speed']:
    data[col] = pd.to_numeric(data[col], errors='coerce')  # Convert to numeric, coerce errors to NaN
    data[col].fillna(data[col].mean(), inplace=True)  # Impute NaN with mean (or other appropriate strateg


# In[24]:


data = data.drop(['Vehicle_Speed', 'Timestamp'], axis=1)


# In[25]:


data['Fraud_indicator'] = data['Fraud_indicator'].map({'Fraud': 1, 'Not Fraud': 0})


# In[26]:


features = ['Transaction_Amount', 'Amount_paid', 'Vehicle_Type', 'Lane_Type', 'Geographical_Location']


# In[27]:


X = data[features]
y = data['Fraud_indicator']


# In[28]:


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[29]:


# Define models
models = {
    'Decision Tree': DecisionTreeClassifier(),
    'Naive Bayes': GaussianNB(),
    'Random Forest': RandomForestClassifier(),
    'XGBoost': XGBClassifier(),
    'Logistic Regression': LogisticRegression(),
    'SVM': SVC()
}


# In[30]:


# Train, predict, and evaluate models
classified_data = pd.DataFrame()  # Dataframe to store classified data


# In[31]:


for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Save the classified data to the dataframe
    classified_data[name + '_Predicted'] = y_pred


# In[32]:


# Concatenate original data with classified data
classified_data = pd.concat([X_test, y_test, classified_data], axis=1)


# In[33]:


# Save the classified data to an Excel file
classified_data.to_excel("classified_data_all_classifiers.xlsx", index=False)


# In[34]:


# Display the dataframe
print("\nClassified Data:")
print(classified_data.head())


# In[35]:


# Train, predict, and evaluate models
results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    confusion = confusion_matrix(y_test, y_pred)
    
    results[name] = {'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'F1 Score': f1, 'Confusion Matrix': confusion}


# In[36]:


label_encoder = LabelEncoder()
data['Vehicle_Type'] = label_encoder.fit_transform(data['Vehicle_Type'])
data['Lane_Type'] = label_encoder.fit_transform(data['Lane_Type'])
data['Geographical_Location'] = label_encoder.fit_transform(data['Geographical_Location'])


# In[37]:


# Preprocess the data, e.g., handle missing values, convert categorical variables to numerical, etc.
# Select relevant features and target variable
# Feature selection
features = ['Transaction_Amount', 'Amount_paid', 'Vehicle_Type', 'Lane_Type', 'Geographical_Location']
X = data[features]
y = data['Fraud_indicator']
# Convert categorical features to numerical using one-hot encoding
X = pd.get_dummies(X, columns=['Transaction_Amount', 'Amount_paid', 'Vehicle_Type', 'Lane_Type', 'Geographical_Location'], drop_first=True)
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Define a list of classifiers
classifiers = [
    LogisticRegression(random_state=42),
    RandomForestClassifier(n_estimators=100, random_state=42),
    GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42),
    SVC(kernel='linear', random_state=42),
    GaussianNB()
]
# Dictionary to store resultsAC
results = {}
# Iterate through classifiers
for clf in classifiers:
    clf_name = clf.__class__.__name__
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    # Store results
    results[clf_name] = {
        'Accuracy': accuracy*100,
        'Precision': precision*100,
        'Recall': recall*100,
        'F1 Score': f1*100
    }
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Fraud', 'Fraud'], yticklabels=['Not Fraud', 'Fraud'])
    plt.title(f'Confusion Matrix - {clf_name}')
    plt.show()
# Display results
results_data = pd.DataFrame(results).T
print(results_data)


# In[38]:


# Dictionary to store results, including train and test accuracy
results = {}

# Iterate through classifiers
for clf in classifiers:
    clf_name = clf.__class__.__name__

    # Fit the model on the training set
    clf.fit(X_train, y_train)

    # Calculate train accuracy
    y_train_pred = clf.predict(X_train)
    train_accuracy = accuracy_score(y_train, y_train_pred)

    # Calculate test accuracy
    y_test_pred = clf.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)

    # Store results, including train and test accuracy
    results[clf_name] = {
        'Train Accuracy': train_accuracy * 100,
        'Test Accuracy': test_accuracy * 100,
        'Precision': precision_score(y_test, y_test_pred) * 100,
        'Recall': recall_score(y_test, y_test_pred) * 100,
        'F1 Score': f1_score(y_test, y_test_pred) * 100
    }

# Display results, including train and test accuracy
results_data = pd.DataFrame(results).T
print(results_data)

