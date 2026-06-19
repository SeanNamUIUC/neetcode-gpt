import torch
import torch.nn as nn
from torchtyping import TensorType

# torch.tensor(python_list) returns a Python list as a tensor
class Solution:
    #we put sentences and divide positive and negative sentences and encode it to number
    def encode_sentences(self, sentences, vocab):
        return [[vocab[word] for word in s.split()] for s in sentences]


    def get_dataset(self, positive: List[str], negative: List[str]) -> TensorType[float]:
        


        #1. combining both positive and negative
        sentences = positive + negative
        #2.  sentences to words
        words = []
        for s in sentences:
            #divide sentences based on whitespace and extend into words list
            words.extend(s.split())
        #3. unique sorted words using set
        unique_words_sorted = sorted(set(words))

        #4. dictionary {number: word}
        vocab = {word: i + 1 for i ,word in enumerate(unique_words_sorted)}

        #5. encoded
        encoded_positive = self.encode_sentences(positive, vocab)
        encoded_negative = self.encode_sentences(negative, vocab)

        # 5) get max length for padding
        max_len = 0
        if encoded_positive:
            max_len = max(max_len, max(len(seq) for seq in encoded_positive))
        if encoded_negative:
            max_len = max(max_len, max(len(seq) for seq in encoded_negative))

        #6. to tensor
        temp_res = encoded_positive + encoded_negative
        # 각 시퀀스를 tensor로 변환
        tensor_seqs = [torch.tensor(seq, dtype=torch.float32) for seq in temp_res]

        # pad_sequence로 길이 맞추기
        padded_res = nn.utils.rnn.pad_sequence(tensor_seqs, batch_first=True, padding_value=0)
        print(padded_res)
        return padded_res

      