from ..feature_scaling.z_score import ZScore
from ..regularizers.l2 import L2
from ..optimizers.gradient_descent import GradientDescent
from ..loss.mse import MSE

import numpy as np

class LinearRegression:
    def __init__(self, loss=None, scaler=None, optimizer=None, lambda_w=0, lambda_b=0, regularization=None, n_iter=10000, convergence_limit=1e-7, learning_rate=0.001):
        # Regularization
        self.lambda_w = lambda_w
        self.lambda_b = lambda_b
        self.regularization = regularization if regularization is not None else L2()

        # Optimizer
        self.optimizer = optimizer if optimizer is not None else GradientDescent(learning_rate=learning_rate, n_iter=n_iter)

        # Scaler
        self.scaler = scaler if scaler is not None else ZScore()

        # Loss
        self.loss = loss if loss is not None else MSE()

        # Convergence Limit
        self.convergence_limit = convergence_limit

    def _model(self, X):
        # Linear Regression Model
        f_wb = np.dot(X, self.weights) + self.bias

        return f_wb

    def _model_gradient(self):
        df_dw = self.X
        df_db = 1

        return df_dw, df_db

    def _cost_function(self):
        R_w = 0
        R_b = 0

        loss_cost = self.loss.cost(self._model(self.X), self.y)

        if self.regularization is not None:
            R_w = (self.lambda_w / self.m) * self.regularization.cost(self.weights)
            R_b = (self.lambda_b / self.m) * self.regularization.cost(self.bias)

        J_wb = loss_cost + R_w + R_b

        return J_wb

    def _gradients(self):
        f_wb = self._model(self.X)

        dL_df = self.loss.gradient(f_wb, self.y)
        df_dw, df_db = self._model_gradient()

        dj_dw = np.dot(dL_df, df_dw)
        dj_db = np.sum(dL_df)

        if self.regularization is not None:
            dj_dw = dj_dw + (self.lambda_w / self.m) * self.regularization.gradient(self.weights)
            dj_db = dj_db + (self.lambda_b / self.m) * self.regularization.gradient(self.bias)

        return dj_dw, dj_db

    def _optimizer(self):
        self.history = []

        for _ in range(self.optimizer.n_iter):
            dj_dw, dj_db = self._gradients()

            self.weights, self.bias = self.optimizer.step(
                [self.weights, self.bias], # Parameters
                [dj_dw, dj_db]             # Gradients
            )

            self.history.append(self._cost_function())

            converged = len(self.history) > 1 and abs(self.history[-1] - self.history[-2]) < self.convergence_limit
            diverged = not np.isfinite(self.history[-1])

            if converged or diverged:
                break

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()

        if self.scaler is not None:
            X = self.scaler.fit(X).transform(X)

        self.X = X
        self.y = y

        self.m = X.shape[0]
        self.n = X.shape[1]

        self.weights = np.zeros(self.n)
        self.bias = 0

        self._optimizer()      

        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)

        if self.scaler is not None:
            X = self.scaler.transform(X)

        y_hat = self._model(X)

        return y_hat

def main():
    ...

if __name__ == "__main__":
    main()