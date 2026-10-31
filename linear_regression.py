import numpy as np

class LinearRegression:
    def __init__(self):
        self._W = None
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
            X: (N, D)
            y: (N, )
        """
        N, D = X.shape
        self._W = np.zeros((D, ))
        cov = X.T @ X # (D, D)
        inv_cov = np.linalg.inv(cov) # (D, D)
        self._W = inv_cov @ X.T @ y # (D, )
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
            X: (N, D)
        """
        y_hat = X @ self._W # (N, )
        
        return y_hat
        
    def log_invert_transform(self, y: np.ndarray):
        ori_y = np.expm1(y)
        
        return ori_y
        
    def evaluate(self, y: np.ndarray, y_hat: np.ndarray):
        y = self.log_invert_transform(y)
        y_hat = self.log_invert_transform(y_hat)
        rmse = ((y - y_hat)**2).mean()**0.5
        
        return rmse
