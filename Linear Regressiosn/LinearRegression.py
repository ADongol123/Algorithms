import numpy as np
from sympy.abc import delta


class LinerRegression:
    def __init__(self,lr = 0.001,iterations = 100,loss='mse',delta=1.0,quantile=0.8):
        """
        Parameters:
        lr: Learning rate
        iterations: Number of iterations of epochs in pytorch
        loss: loss function ('mse','mae',huber','logcosh','quantile')
        delta: Parameters for Huber loss,
        quantile: Quantile for quantile regression
        """

        self.lr = lr
        self.iterations = iterations
        self.loss = loss
        self.delta = delta
        self.quantile = quantile
        self.weights = None
        self.bias = None

    def fit(self,X,y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.iterations):
            y_pred = np.dot(X, self.weights) + self.bias
            errors = y_pred - y

            # Computing gradietns based on loss functions
            if self.loss == "mse":
                dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
                db = (1 / n_samples) * np.sum(errors)

            elif self.loss == "mae":
                # Mean Absolute Error
                dw = (1 / n_samples) * np.dot(X.T, np.sign(errors))
                db = (1 / n_samples) * np.sum(np.sign(errors))


            elif self.loss == 'huber':

                # Huber Loss

                is_small_error = np.abs(errors) <= self.delta

                dw = (1 / n_samples) * np.dot(X.T, np.where(is_small_error, errors, self.delta * np.sign(errors)))

                db = (1 / n_samples) * np.sum(np.where(is_small_error, errors, self.delta * np.sign(errors)))

            elif self.loss == 'logcosh':

                # Log-Cosh Loss

                dw = (1 / n_samples) * np.dot(X.T, np.tanh(errors))
                db = (1 / n_samples) * np.sum(np.tanh(errors))

            elif self.loss == 'quantile':

                # Quantile Loss
                dw = (1 / n_samples) * np.dot(X.T, np.where(errors > 0, self.quantile, -(1 - self.quantile)))
                db = (1 / n_samples) * np.sum(np.where(errors > 0, self.quantile, -(1 - self.quantile)))

            else:
                raise ValueError(f"Unsupported loss function: {self.loss}")





            self.weights = self.weights - self.lr * dw
            self.bias = self.bias - self.lr * db


    def predict(self,X):
        y_pred = np.dot(X, self.weights) + self.bias
        return y_pred
