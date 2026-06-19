import torch
from typing import List, Tuple

class Solution:
    def batch_loader(self, raw_dataset: str, context_length: int, batch_size: int) -> Tuple[List[List[str]]]:
        # You must start by generating batch_size different random indices in the appropriate range
        # using a single call to torch.randint()
        torch.manual_seed(0)
        # will return input => batch_size  * context_length and output => labels
        
        data = list(raw_dataset.split())
        print(data)
        X = []
        Y = []
        indicies = torch.randint(0, (len(data) - context_length), (batch_size,))
        for i in range(batch_size):
            batch_X = []
            batch_Y = []
            for j in range(context_length):
                batch_X.append(data[indicies[i] + j])
                batch_Y.append(data[indicies[i] + j + 1])
            X.append(batch_X)
            Y.append(batch_Y)
       
        res = (X, Y)
        print(res)
        return res


        
