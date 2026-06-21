import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class Solution:
    def backward(self, x: NDArray[np.float64], w: NDArray[np.float64], b: float, y_true: float) -> Tuple[NDArray[np.float64], float]:
        # x: 1D input array
        # w: 1D weight array
        # b: scalar bias
        # y_true: true target value
        #
        # Forward: z = dot(x, w) + b, y_hat = sigmoid(z)
        # Loss: L = 0.5 * (y_hat - y_true)^2
        # Return: (dL_dw rounded to 5 decimals, dL_db rounded to 5 decimals)

        #가중치가 바뀔때 손실이 얼마나 바뀌는가?
        #결과가중치의 의미: 가중치 gradient가 2일때 -> 가중치를 1늘리면 오차가 2늘어남
        #x -> z -> y -> Loss
        #backprop : dl/dw = d1/dy * dy/dz * dz / dw 
        def sigmoid(z):
            return 1 / (1 + np.exp(-z))
        z = np.dot(x, w) + b 
        # print(z)
        y_hat = sigmoid(z)
        # print(y_hat)
        gradient_w = (y_hat - y_true) * y_hat * (1 - y_hat) * x
        gradient_b = (y_hat - y_true) * y_hat * (1 - y_hat)
        return (np.round(gradient_w, 5), np.round(gradient_b, 5))

