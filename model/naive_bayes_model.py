from sklearn.naive_bayes import GaussianNB
import pandas as pd
import numpy as np

def train_model(df):
    # Drop any rows with missing values
    df = df.dropna()

    # Split data into features and target (last column assumed as target)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # Convert all categorical columns to numeric using Label Encoding
    for col in X.columns:
        if X[col].dtype == 'object':
            X[col] = pd.factorize(X[col])[0]

    if y.dtype == 'object':
        y = pd.factorize(y)[0]

    model = GaussianNB()
    model.fit(X, y)
    return model

def classify_instance(model, test_data):
    try:
        prediction = model.predict(test_data)
        return prediction[0]
    except Exception as e:
        return f"Classification Error: {str(e)}"
