from sklearn.svm import OneClassSVM
import numpy as np

class oneclass_svm:

    def __init__(self, nu_value=0.01, kernel='rbf', verbose=False):
        self.model = OneClassSVM(
            nu=nu_value,
            kernel=kernel,
            gamma='scale'
        )

    def fit(self, X):
        self.model.fit(X)

    def predict(self, X):
        preds = self.model.predict(X)
        
        # Convert:
        # +1 → normal (0)
        # -1 → anomaly (1)
        preds = np.where(preds == -1, 1, 0)
        return preds

    def anomaly_ratio(self, X):
        preds = self.predict(X)
        return np.mean(preds)