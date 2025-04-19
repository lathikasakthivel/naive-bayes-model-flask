import pickle
from sklearn.naive_bayes import GaussianNB

def train_model(df):
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    model = GaussianNB()
    model.fit(X, y)

    with open('model/pkl_models/naive_bayes_model.pkl', 'wb') as f:
        pickle.dump(model, f)

    return model

def classify_instance(model, instance):
    return model.predict(instance)[0]
