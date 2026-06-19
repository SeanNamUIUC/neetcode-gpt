import torch
import torch.nn as nn
from torchtyping import TensorType

# 0. Instantiate the linear layers in the following order: Key, Query, Value.
# 1. Biases are not used in Attention, so for all 3 nn.Linear() instances, pass in bias=False.
# 2. torch.transpose(tensor, 1, 2) returns a B x T x A tensor as a B x A x T tensor.
# 3. This function is useful:
#    https://pytorch.org/docs/stable/generated/torch.nn.functional.softmax.html
# 4. Apply the masking to the TxT scores BEFORE calling softmax() so that the future
#    tokens don't get factored in at all.
#    To do this, set the "future" indices to float('-inf') since e^(-infinity) is 0.
# 5. To implement masking, note that in PyTorch, tensor == 0 returns a same-shape tensor 
#    of booleans. Also look into utilizing torch.ones(), torch.tril(), and tensor.masked_fill(),
#    in that order.
class SingleHeadAttention(nn.Module):
    
    def __init__(self, embedding_dim: int, attention_dim: int):
        super().__init__()
        torch.manual_seed(0)
        # 0. Instantiate the linear layers in the following order: Key, Query, Value.
        # 1. Biases are not used in Attention, so for all 3 nn.Linear() instances, pass in bias=False.
        #d_model, d_k
        self.Key = nn.Linear(embedding_dim, attention_dim , bias = False)
        self.Query = nn.Linear(embedding_dim, attention_dim , bias = False)
        self.Value = nn.Linear(embedding_dim, attention_dim , bias = False)
        self.d_model = embedding_dim
        self.d_k = attention_dim
        self.softmax = nn.Softmax(dim=-1)

        # 2. torch.transpose(tensor, 1, 2) returns a B x T x A tensor as a B x A x T tensor.
        # 3. This function is useful:
        #    https://pytorch.org/docs/stable/generated/torch.nn.functional.softmax.html
        # 4. Apply the masking to the TxT scores BEFORE calling softmax() so that the future
        #    tokens don't get factored in at all.
        #    To do this, set the "future" indices to float('-inf') since e^(-infinity) is 0.


    def forward(self, embedded: TensorType[float]) -> TensorType[float]:
        # Return your answer to 4 decimal places

        #0, 1 embedded = (B , T, d_model)
        k = self.Key(embedded) #-> (B, T, d_k)
        q = self.Query(embedded) #-> (B, T, d_k)
        v = self.Value(embedded) #-> (B, T, d_k)

        #T * T score steps
        kT = torch.transpose(k, 1, 2)
        #1. (Q * KT / sqrt(d_k))
        qkT = (torch.matmul(q , kT) / ((self.d_k) ** 0.5))
        #2. apply masking To implement masking, note that in PyTorch, tensor == 0 returns a same-shape tensor 
        # of booleans. Also look into utilizing torch.ones(), torch.tril(), and tensor.masked_fill(),
        # in that order.
        TT = torch.tril(qkT) # changes 0 into upper trianle part
        print("TT is ", TT)
        mask = (TT == 0)
        print("mask is mask",mask)
        TT_masked = TT.masked_fill(mask, float('-inf'))
        print("TT_masked is ",TT_masked)

        #softmax
        softmax_applied = self.softmax(TT_masked)
        print("softmax applied is ", softmax_applied)
        
        # multiply V
        res = torch.matmul(softmax_applied, v)
        return torch.round(res, decimals=4)



      
    






