# Update LM_LSTM module with weight tying, variational dropout

import torch.nn as nn
import torch.nn.functional as F
from functions import *
from torch import Tensor

class VariationalDropout(nn.Module):
    """
    Applies the same dropout mask to recurrent connections within the LSTM by per-
    forming dropout on h_{t-1}
    Each example within the minibatch uses a unique dropout mask, rather than a single
    dropout mask being used over all examples, ensuring diversity in the elements dropped out
    """
    def __init__(self, p=0.0):
        super(VariationalDropout, self).__init__()

        if p < 0.0 or p > 1.0:
            raise ValueError(f"dropout probability has to be between 0 and 1, but got {p}")
        self._p = p

    def _get_mask(self, input):
        # create a ones tensor with the same input tensor shape
        # (batch_size x 1 x embedding_size) will be the size of the mask
        mask = torch.ones((input.size(0), 1, input.size(2)), dtype=input.dtype)
        # create mask
        self._mask = F.dropout(mask, p=self._p, training=self.training)

    def forward(self, input: Tensor) -> Tensor:
        self._get_mask(input)
        # using broadcasting, the mask is replicated for every token in the input sequences
        # Get the device currently in use by PyTorch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return input * self._mask.to(device)
    
class LM_LSTM(nn.Module):
    def __init__(self, emb_size, hidden_size, output_size, pad_index=0, dropout_p=0.1, n_layers=1):
        super(LM_LSTM, self).__init__()
        # Token ids to vectors, we will better see this in the next lab
        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index) # output: (batch_size, vocab_size, emb_size)
        self.dropout = VariationalDropout(p=dropout_p)
        # self.rnns = [nn.LSTM(emb_size, hidden_size) for _ in range(n_layers)]
        self.rnn = nn.LSTM(emb_size, hidden_size)
        self.pad_token = pad_index
        self.nlayers = n_layers

        # Linear layer to project the hidden layer to our output space
        self.output = nn.Linear(hidden_size, output_size) # output: (batch_size, hidden_size, vocab_size)
        # tying weights: weights between the embedding and softmax layer are shared
        self.output.weight = self.embedding.weight
        # print(self.output.weight.shape, self.embedding.weight.shape)


    def forward(self, input_sequence):
        emb = self.embedding(input_sequence)
        # print(f'embedding layer shape={emb.shape}')
        emb = self.dropout(emb)
        output, _ = self.rnn(emb)
        output = self.dropout(output)
        output = self.output(output).permute(0,2,1)

        return output
