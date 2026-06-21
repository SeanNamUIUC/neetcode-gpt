import numpy as np
from numpy.typing import NDArray
from typing import List


class Solution:
    def forward(self, x: NDArray[np.float64], weights: List[NDArray[np.float64]], biases: List[NDArray[np.float64]]) -> NDArray[np.float64]:
        # x: 1D input array
        # weights: list of 2D weight matrices
        # biases: list of 1D bias vectors
        # Apply ReLU after each hidden layer, no activation on output layer
        # return np.round(your_answer, 5)
        def relu(z):
            return np.maximum(z, 0)
        #x : (2,)
        #weights : ()
        
        
        h_before = x.reshape(1 , -1)
        print("shape of x is", x.shape)
        print("shape of h_before is", h_before.shape)
     
       
        for i in range(0, len(weights) - 1):
            print("shape of weight is ", weights[i].shape)
            print("shape of biases is ", biases[i].shape)
            print("shape of h_before is", h_before)
            h_before = relu( h_before @ weights[i]  + biases[i].reshape(1, -1))
           
        print("shape of h is", h_before.shape)
        print("shape of weights -1 is ", weights[-1].shape)
        return np.round( (h_before @ weights[-1] + biases[-1].reshape(1, -1)).flatten(), 5)

