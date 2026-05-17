import numpy as np
from numpy.typing import NDArray


class Solution:

    def binary_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: true labels (0 or 1)
        # y_pred: predicted probabilities
        # Hint: add a small epsilon (1e-7) to y_pred to avoid log(0)
        # return round(your_answer, 4)
        loss = 0
        
        for i in range(len(y_true)):
            epsilon = y_pred[i] + 0.0000001
            loss = loss + ((y_true[i] * np.log(epsilon)) + ((1-y_true[i]) * (np.log(1-epsilon))))
        loss = (loss/len(y_true)) * -1
        return np.round(loss, 4)

    def categorical_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: one-hot encoded true labels (shape: n_samples x n_classes)
        # y_pred: predicted probabilities (shape: n_samples x n_classes)
        # Hint: add a small epsilon (1e-7) to y_pred to avoid log(0)
        # return round(your_answer, 4)
        loss = 0
        epsilon = 0.0000001
        for i in range(len(y_true)):
            for j in range(len(y_true[0])):
                loss = loss + (y_true[i][j]) * np.log(y_pred[i][j]+epsilon)
        loss = (loss/len(y_true)) * -1
        return np.round(loss, 4)
