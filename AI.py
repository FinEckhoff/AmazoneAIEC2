from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import pandas as pd
import numpy as np



svc = SVC(kernel='rbf')
svc.fit(X_train, y_train)
y_pred = svc.predict(X_test)
print(accuracy_score(y_test, y_pred))