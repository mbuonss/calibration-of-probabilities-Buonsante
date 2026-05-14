import numpy as np


class LogisticRegression:

    def __init__(self, lr = 0.1, num_iter = 1000, tol = 1e-7):
        self.lr = lr
        self.num_iter = num_iter
        self.tol = tol
        self.theta = None

    def _add_bias(self, X):
        return np.c_[np.ones((X.shape[0], 1)), X]

    def fit(self, X, y):
        X_bias = self._add_bias(X)

        n_samples, n_features = X_bias.shape
        self.theta = np.zeros(n_features)

        for _ in range(self.num_iter):
            # avanti
            z = X_bias @ self.theta
            y_pred = self._sigmoid(z)

            # gradiente
            grad = (X_bias.T @ (y_pred - y)) / n_samples

            # aggiornamento
            self.theta -= self.lr * grad

            # condizione di stoè
            if np.linalg.norm(grad) < self.tol:
                break

    def _sigmoid(self, z):
        z = np.clip(z, -500, 500) #per evitare overflow
        return 1.0 / (1.0 + np.exp(-z))

    def pred_probab(self, X):
        X_bias = self._add_bias(X)
        return self._sigmoid(X_bias @ self.theta)

    def predict(self, X, threshold=0.5):
        probab = self.pred_probab(X)
        return (probab >= threshold).astype(int)