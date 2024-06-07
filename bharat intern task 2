import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import accuracy_score

# Function to load the data
def load_data(uploaded_file):
    df = pd.read_csv(uploaded_file)
    return df

# Function to preprocess the data
def preprocess_data(df):
    df['Embarked'].fillna('S', inplace=True)
    df['Age'].fillna(df.Age.median(), inplace=True)
    df.drop('Cabin', axis=1, inplace=True)
    df['Sex'] = df['Sex'].map({'male': 1, 'female': 0})
    df = df[df['Embarked'] != 'Q']
    df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1})
    df.drop(['PassengerId', 'Name', 'Ticket'], axis=1, inplace=True)
    return df

# Function to visualize the data
def visualize_data(df):
    st.subheader('Survival Counts')
    fig, ax = plt.subplots()
    sns.countplot(x='Survived', data=df, ax=ax)
    st.pyplot(fig)

    st.subheader('Survival Counts by Sex')
    fig, ax = plt.subplots()
    sns.countplot(x='Survived', hue='Sex', data=df, ax=ax)
    st.pyplot(fig)

    st.subheader('Survival Counts by Pclass')
    fig, ax = plt.subplots()
    sns.countplot(x='Survived', hue='Pclass', data=df, ax=ax)
    st.pyplot(fig)

    st.subheader('Age Distribution')
    fig, ax = plt.subplots()
    sns.histplot(x='Age', data=df, kde=True, ax=ax)
    st.pyplot(fig)

# Function to train and evaluate the model
def train_and_evaluate(df):
    X = df[['Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Pclass']]
    y = df['Survived']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)
    reg = LogisticRegression()
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)

    comparison_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    st.subheader("Comparison of Actual and Predicted Values")
    st.write(comparison_df.head(10))

    error = metrics.mean_absolute_error(y_test, y_pred)
    st.write(f'The Mean Absolute Error is: {error * 100:.2f}%')

    score = accuracy_score(y_pred, y_test)
    st.write(f"The Accuracy Score is: {score * 100:.2f}%")

# Streamlit UI
st.title('Titanic Survival Prediction')

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
if uploaded_file is not None:
    df = load_data(uploaded_file)
    
    st.subheader('Raw Data')
    st.write(df.head())
    
    df = preprocess_data(df)
    
    st.subheader('Preprocessed Data')
    st.write(df.head())
    
    visualize_data(df)
    
    train_and_evaluate(df)
