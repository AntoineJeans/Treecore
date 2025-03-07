import numpy as np

def predictions_to_y(predictions):
    return np.argmax(predictions, axis=1)

from sklearn.metrics import accuracy_score
def get_accuracy_score(model, X, y, X_test, y_test):
    model.fit(X, y)
    y_pred = model.predict(X_test)
    return accuracy_score(y_test, y_pred)