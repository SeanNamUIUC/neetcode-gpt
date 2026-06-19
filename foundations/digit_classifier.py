import torch
import torch.nn as nn
from torchtyping import TensorType

class Solution(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        # Define the architecture here
        self.first_layer = nn.Linear(28 * 28, 512)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        self.final_layer = nn.Linear(512, 10)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, images: TensorType[float]) -> TensorType[float]:
        torch.manual_seed(0)
        
        # Return the model's prediction to 4 decimal places
        first_layer_output = self.first_layer(images)
        activated_relu = self.relu(first_layer_output)
        activated_dropout = self.dropout(activated_relu)
        final_layer_output = self.final_layer(activated_dropout)
        activated_sigmoid = self.sigmoid(final_layer_output)
        return torch.round(activated_sigmoid, decimals = 4)

        


