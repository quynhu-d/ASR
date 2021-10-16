from torch import nn
from torch.nn import Sequential

from hw_asr.base import BaseModel


class LSTMModel(BaseModel):
    def __init__(self, n_feats, n_class, fc_hidden=512, *args, **kwargs):
        super().__init__(n_feats, n_class, *args, **kwargs)
        self.rnn = nn.LSTM(input_size=n_feats, hidden_size=fc_hidden, batch_first=True)
        self.fc_hidden = fc_hidden
        self.n_feats = n_feats
        self.clf = nn.Linear(fc_hidden, n_class)

    def forward(self, spectrogram, spectrogram_length, *args, **kwargs):
        padded_input = nn.utils.rnn.pack_padded_sequence(
            spectrogram, spectrogram_length, batch_first=True, enforce_sorted=False)
        output, _ = self.rnn(padded_input)

        # learn more about gather
        # https://medium.com/analytics-vidhya/understanding-indexing-with-pytorch-gather-33717a84ebc4
        # last_hidden = torch.gather(
        #     output,
        #     dim=1,
        #     # substract 1 because index start from 0 (not 1 as length)
        #     index=spectrogram_length.sub(1).view(-1, 1, 1).expand(-1, -1, self.fc_hidden)
        # )
        # print('last_hidden shape', last_hidden.shape)

        logits = self.clf(nn.utils.rnn.pad_packed_sequence(output, batch_first=True)[0])
        return {"logits": logits}

    def transform_input_lengths(self, input_lengths):
        return input_lengths  # we don't reduce time dimension here
