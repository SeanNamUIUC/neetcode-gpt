import torch
import torch.nn as nn
from torchtyping import TensorType

class Solution(nn.Module):
    def __init__(self, vocabulary_size: int):
        super().__init__()
        torch.manual_seed(0)  # 랜덤 시드를 고정시켜서 결과를 재현 가능하게 만듦
        self.embedding_layer = nn.Embedding(vocabulary_size, 16)  # 임베딩 레이어 (vocab_size x 16)
        self.Linear_layer = nn.Linear(16, 1)  # 선형 레이어 (16 -> 1)
        self.Sigmoid = nn.Sigmoid()  # Sigmoid 활성화 함수

    def forward(self, x: TensorType[int]) -> TensorType[float]:
        """
        입력 텐서 x는 (B, T) 형태: B는 배치 크기, T는 텍스트의 단어 수 (각 샘플의 길이)
        """
        #using embedding layer (C) as 16
        #averaging is called "Bag of words"
        # Hint: The embedding layer outputs a B, T, embed_dim tensor
        # but you should average it into a B, embed_dim tensor before using the Linear layer

        # Return a B, 1 tensor and round to 4 decimal places

        # 1. 임베딩 레이어를 통과한 후 (B, T) -> (B, T, embed_dim)
        embedding_output = self.embedding_layer(x)  # (B, T) -> (B, T, 16)
        # 2. 평균값을 내서 T 차원(단어 수)을 없애고 (B, 16) 형태로 만듦
        averaging_output = torch.mean(embedding_output, dim=1)  # (B, T, 16) -> (B, 16)
        # 3. 선형 레이어를 통과시켜 (B, 1) 형태로 만듦
        linear_output = self.Linear_layer(averaging_output)  # (B, 16) -> (B, 1)
        # 4. Sigmoid 활성화 함수를 적용하여 결과를 0과 1 사이로 변환
        activate_sigmoid = self.Sigmoid(linear_output)  # (B, 1) -> (B, 1) (0~1 사이 값)
        # 5. 소수점 4자리까지 반올림
        return torch.round(activate_sigmoid , decimals=4) 

