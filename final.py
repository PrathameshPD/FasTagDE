#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df = pd.read_csv("C:\\Users\\Panda\\Desktop\\Fast_TagDE.csv")


# In[3]:


df.head()


# In[4]:


df.tail()


# In[5]:


df.info()


# In[6]:


df.describe()


# In[7]:


df.shape


# In[8]:


df.size


# In[9]:


df.columns


# In[10]:


df.isnull().sum()


# In[11]:


FTG = 'FastagID'
# 1. Fill missing values with a specific value (e.g., 0)
df[FTG] = df[FTG].fillna(0)


# In[12]:


df.isnull().sum()


# In[13]:


df.nunique()


# In[14]:


import matplotlib.pyplot as plt


# In[15]:


import seaborn as sns


# In[16]:


# Count plot for Vehicle Type
plt.figure(figsize=(10, 6))
sns.countplot(x='Vehicle_Type', data=df)
plt.title('Count of Transactions by Vehicle Type')
plt.xlabel('Vehicle Type')
plt.ylabel('Transaction Count')
plt.show()


# In[17]:


#Countplot for Lane Type
plt.figure(figsize=(8, 5))
sns.countplot(x='Lane_Type', data=df, palette='pastel')
plt.title('Distribution of Lane Types')
plt.xlabel('Lane Type')
plt.ylabel('Count')
plt.show()


# In[18]:


# Visualize the distribution of Amount using a histogram
plt.figure(figsize=(10, 6))
sns.histplot(df['Amount_paid'], bins=30, kde=True, color='blue')
plt.title('Distribution of Amount_paid')
plt.xlabel('Amount paid')
plt.ylabel('Frequency')
plt.show()


# In[19]:


# Toll booth activity distribution
plt.figure(figsize=(10, 6))
df.groupby("TollBoothID").agg(count=("Transaction_ID", "count"), revenue=("Transaction_Amount", "sum")).plot(kind="bar", stacked=True)
plt.title("Toll Booth Activity Distribution")
plt.xlabel("Toll Booth ID")
plt.ylabel("Count (left axis), Revenue (right axis)")
plt.show()

# Toll booth heatmap
plt.figure(figsize=(10, 6))
pivot_table = pd.pivot_table(df, values="Transaction_ID", index="Vehicle_Type", columns="TollBoothID", aggfunc="count")
sns.heatmap(pivot_table, cmap="YlGnBu", annot=True)
plt.title("Toll Booth Activity Heatmap by Vehicle Type")
plt.show()


# In[20]:


# Lane usage comparison
plt.figure(figsize=(10, 6))
df.groupby("Lane_Type").agg(count=("Transaction_ID", "count"), mean_amount=("Transaction_Amount", "mean")).plot(kind="bar")
plt.title("Lane Usage Comparison")
plt.xlabel("Lane Type")
plt.ylabel("Count (left axis), Average Amount (right axis)")
plt.show()

# Boxplot of transaction amount by lane type
plt.figure(figsize=(10, 6))
sns.boxplot(x="Lane_Type", y="Transaction_Amount", showmeans=True, data=df
           )
plt.title("Distribution of Transaction Amount by Lane Type")
plt.show()


# In[21]:


# Vehicle speed distribution by vehicle type
plt.figure(figsize=(10, 6))
sns.violinplot(x="Vehicle_Type", y="Vehicle_Speed", data=df)
plt.title("Distribution of Vehicle Speed by Vehicle Type")
plt.show()



# In[22]:


# Boxplot of transaction amount by fraud indicator
plt.figure(figsize=(10, 6))
sns.boxplot(x="Fraud_indicator", y="Transaction_Amount", showmeans=True, data=df)
plt.title("Distribution of Transaction Amount by Fraud Indicator")
plt.show()


# In[23]:


df['is_duplicate_fastag'] = df.duplicated(subset='FastagID', keep=False)
df['is_duplicate_plate'] = df.duplicated(subset='Vehicle_Plate_Number', keep=False)


# In[24]:


label_encoder = LabelEncoder()
df['Vehicle_Type'] = label_encoder.fit_transform(df['Vehicle_Type'])
df['Lane_Type'] = label_encoder.fit_transform(df['Lane_Type'])
df['Geographical_Location'] = label_encoder.fit_transform(df['Geographical_Location'])

for col in ['Transaction_Amount', 'Amount_paid', 'Vehicle_Speed']:
    df[col] = pd.to_numeric(df[col], errors='coerce')  # Convert to numeric, coerce errors to NaN
    df[col].fillna(df[col].mean(), inplace=True)  # Impute NaN with mean (or other appropriate strateg


# In[25]:


df = df.drop(['Vehicle_Speed', 'Timestamp'], axis=1)


# In[26]:


df['Fraud_indicator'] = df['Fraud_indicator'].map({'Fraud': 1, 'Not Fraud': 0})


# In[27]:


features = ['Transaction_Amount', 'Amount_paid', 'Vehicle_Type', 'Lane_Type', 'Geographical_Location']


# In[28]:


X = df[features]
y = df['Fraud_indicator']


# In[29]:


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[30]:


# Define models
models = {
    'Decision Tree': DecisionTreeClassifier(),
    'Naive Bayes': GaussianNB(),
    'Random Forest': RandomForestClassifier(),
    'XGBoost': XGBClassifier(),
    'Logistic Regression': LogisticRegression(),
    'SVM': SVC()
}


# In[31]:


# Train, predict, and evaluate models
classified_data = pd.DataFrame()  # Dataframe to store classified data


# In[32]:


for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Save the classified data to the dataframe
    classified_data[name + '_Predicted'] = y_pred


# In[33]:


# Concatenate original data with classified data
classified_data = pd.concat([X_test, y_test, classified_data], axis=1)


# In[34]:


# Save the classified data to an Excel file
classified_data.to_excel("classified_data_all_classifiers.xlsx", index=False)


# In[35]:


# Display the dataframe
print("\nClassified Data:")
print(classified_data.head())


# In[36]:


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


# In[37]:


label_encoder = LabelEncoder()
df['Vehicle_Type'] = label_encoder.fit_transform(df['Vehicle_Type'])
df['Lane_Type'] = label_encoder.fit_transform(df['Lane_Type'])
df['Geographical_Location'] = label_encoder.fit_transform(df['Geographical_Location'])


# In[38]:


# Preprocess the data, e.g., handle missing values, convert categorical variables to numerical, etc.
# Select relevant features and target variable
# Feature selection
features = ['Transaction_Amount', 'Amount_paid', 'Vehicle_Type', 'Lane_Type', 'Geographical_Location']
X = df[features]
y = df['Fraud_indicator']
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
# Dictionary to store results
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


# In[39]:


# ... (previous code)

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

    # ... (rest of your code for confusion matrix and visualization)

# Display results, including train and test accuracy
results_data = pd.DataFrame(results).T
print(results_data)


# In[ ]:





# In[ ]:





# In[ ]:




