import numpy as np
from util import binary_cross_entropy
def sigmoid(x):
    return 1/(1+np.exp(-x))


class LogisticRegression:
    def __init__(self, lr=0.1, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y,loss_function = binary_cross_entropy):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
		self.loss_history = []
		
        for _ in range(self.n_iters):
            linear_pred = np.dot(X, self.weights) + self.bias
            predictions = sigmoid(linear_pred)
			
            # This part is the gradient descent 
            dw = (1/n_samples) * np.dot(X.T, (predictions - y))
            db = (1/n_samples) * np.sum(predictions-y)

            self.weights = self.weights - self.lr * dw
            self.bias = self.bias - self.lr * db


    def predict(self, X):
        linear_pred = np.dot(X, self.weights) + self.bias
        y_pred = sigmoid(linear_pred)
        # Giving the result zero and one
        # if the estimation is going to be between 0 and 0.5 it will return 0 else it will return 1
        class_pred = [0 if y<=0.5 else 1 for y in y_pred]
        return class_pred