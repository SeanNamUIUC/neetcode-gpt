import numpy as np
from numpy.typing import NDArray


class Solution:
    def forward(self, x: NDArray[np.float64], gamma: NDArray[np.float64], beta: NDArray[np.float64]) -> NDArray[np.float64]:
        # x: 1D feature vector
        # gamma: 1D scale parameter (same length as x)
        # beta: 1D shift parameter (same length as x)
        # eps = 1e-5
        # Normalize: x_hat = (x - mean) / sqrt(var + eps)
        # Scale and shift: out = gamma * x_hat + beta
        # return np.round(your_answer, 5)


        #To 인공지능 모델이 깊어지면, 앞쪽 가중치 나사가 아주 조금만 바뀌어도 
        #뒤쪽 레이어로 갈수록 그 파도가 엄청나게 커져서 값이 수천, 수만으로 튀거나 소멸해서 이를 막기위해
        #mean -> 0 , sigma -> 1
        #batchnorm -> Batch Normalization normalizes each features independently across its samples: 특징들의 평균,표편 
        #layer_norm -> Layer Normalization normalizes each sample independently across its features.: samples들의 평균, 표편
        mean = np.mean(x)
        var = np.mean(np.square(x - mean))
        Normalized = ((x - mean) / np.sqrt(var + 1e-5))
        scale_and_shift = gamma * Normalized + beta
        return np.round(scale_and_shift, 5)


