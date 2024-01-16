from turtle import pd
from bs4 import ResultSet
import streamlit as st
from final import df, plt
import matplotlib.pyplot as plt
import seaborn as sns
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


# find more emojis here:// https://www.webfx.com/tools/emoji-cheat-sheet/
st.set_page_config(page_title="FASTAG", page_icon=":tada:")



st.subheader("MINOR PROJECT")
st.write("---")
st.write("Under the Guidance of : ")
st.write("Dr. S. Senthil")
st.write("Director School of CSA")
st.write("---")
st.write("Prathamesh - 041 / Nikhil - 035")
st.write("---")
st.title("Enchancing Integrity of Toll Gate Revenue: Fastag Fraud Detection")
st.write("Toll gates are essential for controlling traffic volume and funding road upkeep and expansion in this age of expanding infrastructure and developing technologies. Fast lag technology has been widely used in toll transactions, increasing their efficiency, and facilitating smooth, cashless transactions. The possibility of fraud in toll gate transactions, particularly regarding the misrepresentation of vehicle types, has emerged as a new problem because of this achievement."
"To calculate the proper toll fees, vehicles such as XUVs, sedans, and compact cars must be categorized. The accuracy of vehicle classification must be guaranteed as the use of Fast lag transactions grows toprevent fraudulent attempts in which cars fraudulently claim to be in a lesser category to avoid paying higher tolls.")

with st.container():
    st.write("---")
    st.write("Top 5 Dataset")
    st.dataframe(df.head())

with st.container():
    st.write("---")
    left_column, right_column = st.columns(2)
    with left_column:
       st.header("Scope of the Problem")
       st.write('##')
       st.write("The scope of the problem revolves around the accurate categorization of vehicles at toll gates to ensure fair and appropriate toll charges. The issue is particularly prominent in the context of Fast lag transactions, where vehicles may attempt to misrepresent their category to pay a lower toll fee. The problem encompasses the need for a robust system that can reliably identify whether a vehicle is an XUV, Sedan, or Mini Car during FastTag transactions.")
       st.write("Key aspects of the problem include:")
       st.write("1. Categorization Accuracy")
       st.write("2. Fraud Detection")

with st.container():
    st.write("---")
    st.write("visualizes the count of transactions by vehicle type using a countplot.")
    plt.figure(figsize=(10, 6))
    sns.countplot(x='Vehicle_Type', data=df)
    plt.title('Count of Transactions by Vehicle Type')
    plt.xlabel('Vehicle Type')
    plt.ylabel('Transaction Count')
    st.pyplot()
    st.set_option('deprecation.showPyplotGlobalUse', False)

with st.container():
    st.write("---")
    st.write("visualizes the distribution of lane types using a countplot")
    plt.figure(figsize=(8, 5))
    sns.countplot(x='Lane_Type', data=df, palette='pastel')
    plt.title('Distribution of Lane Types')
    plt.xlabel('Lane Type')
    plt.ylabel('Count')
    st.pyplot()
    st.set_option('deprecation.showPyplotGlobalUse', False)

with st.container():
    st.write("---")
    st.write("visualizes the distribution of transaction amount using a histogram")
    plt.figure(figsize=(10, 6))
    sns.histplot(df['Amount_paid'], bins=30, kde=True, color='blue')
    plt.title('Distribution of Amount_paid')
    plt.xlabel('Amount paid')
    plt.ylabel('Frequency')
    st.pyplot()
    st.set_option('deprecation.showPyplotGlobalUse', False)

with st.container():
    st.write("---")
    st.write("Toll Booth Activity: Visualize the distribution of transactions and total revenue across different toll booths using bar charts or heatmaps. This can reveal busy locations and potential revenue hotspots.")
    plt.figure(figsize=(10, 6))
    df.groupby("TollBoothID").agg(count=("Transaction_ID", "count"), revenue=("Transaction_Amount", "sum")).plot(kind="bar", stacked=True)
    plt.title("Toll Booth Activity Distribution")
    plt.xlabel("Toll Booth ID")
    plt.ylabel("Count (left axis), Revenue (right axis)")
    st.pyplot()
    st.set_option('deprecation.showPyplotGlobalUse', False)

with st.container():
    st.write("---")
    st.write("Lane Usage Comparison: Compare the distribution of transactions and average transaction amount across different lane types (ETC vs Manual) using bar charts or boxplots. This can highlight potential efficiency differences.")
    # Lane usage comparison
    plt.figure(figsize=(10, 6))
    df.groupby("Lane_Type").agg(count=("Transaction_ID", "count"), mean_amount=("Transaction_Amount", "mean")).plot(kind="bar")
    plt.title("Lane Usage Comparison")
    plt.xlabel("Lane Type")
    plt.ylabel("Count (left axis), Average Amount (right axis)")
    st.pyplot()
    st.set_option('deprecation.showPyplotGlobalUse', False)

with st.container():
    st.write("---")
    st.write("visualizes the distribution of transaction amount for different lane types using a boxplot.")
    # Boxplot of transaction amount by lane type
    plt.figure(figsize=(10, 6))
    sns.boxplot(x="Lane_Type", y="Transaction_Amount", showmeans=True, data=df)
    plt.title("Distribution of Transaction Amount by Lane Type")
    st.pyplot()
    st.set_option('deprecation.showPyplotGlobalUse', False)

with st.container():
    st.write("---")
    st.write("visualizes the distribution of transaction amount for fraudulent and non-fraudulent transactions using a boxplot.")
    # Boxplot of transaction amount by fraud indicator
    plt.figure(figsize=(10, 6))
    sns.boxplot(x="Fraud_indicator", y="Transaction_Amount", showmeans=True, data=df)
    plt.title("Distribution of Transaction Amount by Fraud Indicator")
    st.pyplot()
    st.set_option('deprecation.showPyplotGlobalUse', False)

with st.container():
    st.write("---")
    st.write("Confusion Matrix")
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
    cm_plot = st.pyplot()
# Create results table
results_data = pd.DataFrame(results).T

# Display results on the Streamlit app
st.header("Model Performance Results")
st.table(results_data)







