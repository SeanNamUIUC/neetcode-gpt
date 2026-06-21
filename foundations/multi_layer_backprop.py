import numpy as np
from typing import List

class Solution:
    def forward_and_backward(self,
                             x: List[float],
                             W1: List[List[float]], b1: List[float],
                             W2: List[List[float]], b2: List[float],
                             y_true: List[float]) -> dict:
        
        # 1. 활성화 함수 및 미분 정의
        def relu(z):
            return np.maximum(z, 0)
        def relu_prime(z):
            return np.where(z > 0, 1, 0)
            

        # .reshape(-1, 1) to fix the rule
        x = np.array(x).reshape(-1, 1)          # (2, 1)
        W1 = np.array(W1)                        # (2, 2)
        b1 = np.array(b1).reshape(-1, 1)        # (2, 1)
        W2 = np.array(W2)                        # (1, 2)
        b2 = np.array(b2).reshape(-1, 1)        # (1, 1)
        y_true = np.array(y_true).reshape(-1, 1) # (1, 1)

        #Forward Pass 
        z1 = W1 @ x + b1                         # (2, 2) @ (2, 1) + (2, 1) -> (2, 1)
        a1 = relu(z1)                            # (2, 1)
        z2 = W2 @ a1 + b2                        # (1, 2) @ (2, 1) + (1, 1) -> (1, 1)
        
        # compute Loss
        Loss = np.mean(np.square(z2 - y_true))

        # Backward Pass
        # 1. 출력층 오차
        dz2 = 2 * (z2 - y_true)                        # (1, 1)

        # 2. W2, b2 책임 수거 (W2 짝꿍인 a1이 뒤집혀서 뒤에 붙음)
        dw2 = dz2 @ a1.T                         # (1, 1) @ (1, 2) -> (1, 2) [same size with W2]
        db2 = dz2                                # (1, 1)

        # 3. 에러 역류 (가중치 통로 W2가 뒤집혀서 앞에 붙음)
        da1 = W2.T @ dz2                         # (2, 1) @ (1, 1) -> (2, 1)

        # 4. ReLU 필터 통과
        dz1 = da1 * relu_prime(z1)               # (2, 1) * (2, 1) -> (2, 1)

        # 5. W1, b1 책임 수거 (W1 짝꿍인 x가 뒤집혀서 뒤에 붙음)
        dw1 = dz1 @ x.T                          # (2, 1) @ (1, 2) -> (2, 2) [원래 W1 크기 일치!]
        db1 = dz1                                # (2, 1)

        # 🏁 [출력 양식 맞추기] 문제 조건에 맞춰 차원 복구 및 리스트화
        return {
            'loss': round(float(Loss), 4),
            
            # dW1, dW2는 원래 2D 행렬이었으니 .tolist()하면 자동으로 2D list
            'dW1': np.round(dw1, 4).tolist(),
            'dW2': np.round(dw2, 4).tolist(),
            
            # db1, db2는 원래 1차원 리스트였으니 .flatten()으로 납작하게 편 뒤 .tolist()
            # 이렇게 하면 db2도 원소가 1개인 [1.0] 형태의 1차원 리스트로 깔끔하게 떨어집니다.
            'db1': np.round(db1.flatten(), 4).tolist(),
            'db2': np.round(db2.flatten(), 4).tolist()
        }
