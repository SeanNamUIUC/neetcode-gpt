import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array of logits
        # Hint: subtract max(z) for numerical stability before computing exp
        # return np.round(your_answer, 4)
        max_val = max(z)
        for i in range(0 , len(z)):
            z[i] = np.exp(z[i] - max_val)
        sum = np.sum(z)
        res = z / sum
        return np.round(res, 4)
