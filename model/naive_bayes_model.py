from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np

# To store encoders for each column so we can transform new instances consistently
encoders = {}

def train_model(df):
    global encoders
    encoders = {}

    # Drop missing values
    df = df.dropna()

    # Split into features and target
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # Encode features (both strings and numbers)
    for col in X.columns:
        if X[col].dtype == 'object':
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])
            encoders[col] = le

    # Encode target if it's string
    if y.dtype == 'object':
        le_target = LabelEncoder()
        y = le_target.fit_transform(y)
        encoders['target'] = le_target

    # Train model
    model = GaussianNB()
    model.fit(X, y)

    return model

def classify_instance(model, instance_data):
    global encoders

    try:
        # Check if we have encoders for any string columns
        processed_data = []
        for i, col in enumerate(encoders.keys()):
            if col == 'target':
                continue
            value = instance_data[0][i]
            # Try to convert to float, if fails — assume it's string and encode
            try:
                processed_data.append(float(value))
            except:
                le = encoders.get(col)
                if le and value in le.classes_:
                    processed_data.append(le.transform([value])[0])
                else:
                    return f"Error: Unknown value '{value}' for feature '{col}'"

        processed_data = np.array([processed_data])

        # Predict
        prediction = model.predict(processed_data)

        # Decode target if encoded
        if 'target' in encoders:
            prediction = encoders['target'].inverse_transform(prediction)

        return prediction[0]

    except Exception as e:
        return f"Classification Error: {str(e)}"
