from torch import nn
from torch.nn import Sequential

from hw_asr.base import BaseModel


class LSTMModel(BaseModel):
    def __init__(self, n_feats, n_class, fc_hidden=512, *args, **kwargs):
        super().__init__(n_feats, n_class, *args, **kwargs)
        self.rnn = nn.LSTM(input_size=n_feats, hidden_size=fc_hidden, batch_first=True)
        self.fc_hidden = fc_hidden
        self.n_feats = n_feats
        self.clf = Sequential(
            nn.Linear(fc_hidden, fc_hidden),
            nn.ReLU(),
            nn.Linear(fc_hidden, n_class)
        )

    def forward(self, spectrogram, spectrogram_length, *args, **kwargs):

        output, _ = self.rnn(spectrogram)

        # Now we want to take the last hidden state of each instance in batch
        # BUT we don't want to take `padding` hidden state
        # We will use `torch.gather` and `length` to do that

        # learn more about gather
        # https://medium.com/analytics-vidhya/understanding-indexing-with-pytorch-gather-33717a84ebc4
        # last_hidden = torch.gather(
        #     output,
        #     dim=1,
        #     # substract 1 because index start from 0 (not 1 as length)
        #     index=spectrogram_length.sub(1).view(-1, 1, 1).expand(-1, -1, self.fc_hidden)
        # )
        # print('last_hidden shape', last_hidden.shape)

        logits = self.clf(output)
        return {"logits": logits}

    def transform_input_lengths(self, input_lengths):
        return input_lengths  # we don't reduce time dimension here
