import numpy as np
from typing import Tuple, List


class Solution:
    def batch_norm(self, x: List[List[float]], gamma: List[float], beta: List[float],
                   running_mean: List[float], running_var: List[float],
                   momentum: float, eps: float, training: bool) -> Tuple[List[List[float]], List[float], List[float]]:
        # During training: normalize using batch statistics, then update running stats
        # During inference: normalize using running stats (no batch stats needed)
        # Apply affine transform: y = gamma * x_hat + beta
        # Return (y, running_mean, running_var), all rounded to 4 decimals as lists
        # 1. 모든 입력을 넘파이 배열로 안전하게 변환
        x = np.array(x)
        gamma = np.array(gamma)
        beta = np.array(beta)
        running_mean = np.array(running_mean)
        running_var = np.array(running_var)
        
        if training:
            #broadcasting automatically eventhough matrix size is different
            #batch_statistics
            mean_batch = np.mean(x, axis=0)
            var_batch = np.mean(np.square(x - mean_batch), axis=0)
            
            # normalization
            x_hat = (x - mean_batch) / np.sqrt(var_batch + eps)
            
            #running stats update
            running_mean = (1 - momentum) * running_mean + momentum * mean_batch
            running_var = (1 - momentum) * running_var + momentum * var_batch
        else:
            # 3.inference mode, using original running stats
            x_hat = (x - running_mean) / np.sqrt(running_var + eps)
            
        y = gamma * x_hat + beta
        
        # 5.tolist
        return (
            np.round(y, 4).tolist(),
            np.round(running_mean, 4).tolist(),
            np.round(running_var, 4).tolist()
        )


        


