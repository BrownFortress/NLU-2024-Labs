# 1. Replace RNN with a LSTM
# 2. Add two dropout layers

import torch.nn as nn

class LM_LSTM(nn.Module):
    def __init__(self, emb_size, hidden_size, output_size, pad_index=0, out_dropout=0.1,
                 emb_dropout=0.1, n_layers=1):
        super(LM_LSTM, self).__init__()
        # Token ids to vectors, we will better see this in the next lab
        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)
        self.emb_dropout = nn.Dropout(emb_dropout)
        self.lstm = nn.LSTM(emb_size, hidden_size, n_layers, bidirectional=False, batch_first=True)
        self.pad_token = pad_index
        self.linear_dropout = nn.Dropout(out_dropout)
        # Linear layer to project the hidden layer to our output space
        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, input_sequence):
        emb = self.embedding(input_sequence)
        emb_drop = self.emb_dropout(emb)
        lstm_out, _  = self.lstm(emb_drop)
        linear_drop = self.linear_dropout(lstm_out)
        output = self.output(linear_drop).permute(0,2,1)
        return output