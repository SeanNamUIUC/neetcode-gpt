import numpy as np
from numpy.typing import NDArray


class Solution:

    def binary_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        #모델이 틀릴수록 더 큰 벌점을 가함
        # y_true: true labels (0 or 1)
        # y_pred: predicted probabilities
        # Hint: add a small epsilon (1e-7) to y_pred to avoid log(0)
        # return round(your_answer, 4)
        y_pred += 1e-7
        sum = 0
        for i in range(0, len(y_true)):
            sum += y_true[i] * np.log(y_pred[i]) + (1 - y_true[i]) * np.log(1 - y_pred[i])
        return round (-1 * sum / len(y_true), 4)
        

    def categorical_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: one-hot encoded true labels (shape: n_samples x n_classes)
        # y_pred: predicted probabilities (shape: n_samples x n_classes)
        # Hint: add a small epsilon (1e-7) to y_pred to avoid log(0)
        # return round(your_answer, 4)
        sum = 0
        for i in range(0, len(y_true)):
            for j in range(0, len(y_true[0])):
                sum += y_true[i][j] * np.log(y_pred[i][j])
        return round (-1 * sum / len(y_true), 4)
                
