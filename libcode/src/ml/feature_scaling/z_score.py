from .base import Scaler

import numpy as np


class ZScore(Scaler):
    """
    z = (x - mean) / std
    """

    def fit(self, X):
        X = np.asarray(X, dtype=float)

        self.mean = X.mean(axis=0)
        
        std = X.std(axis=0)
        self.std = np.where(std == 0, 1.0, std)

        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)

        X_scaled = (X - self.mean) / self.std

        return X_scaled