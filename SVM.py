from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split

svc = SVC(kernel='rbf')


def run():
    df = \
        pd.read_csv(
            'https://raw.githubusercontent.com/Raghavagr/Student-Placement-Predictor/main/students_placement.csv')

    X = df.drop(columns=['placed'])
    y = df['placed']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

    svc.fit(X_train, y_train)
    y_pred = svc.predict(X_test)
    return accuracy_score(y_test, y_pred)


def save():
    pickle.dump(svc, open('model.pkl', 'wb'))
