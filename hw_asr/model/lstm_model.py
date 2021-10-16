from torch import nn
from torch.nn import Sequential

from hw_asr.base import BaseModel


class LSTMModel(BaseModel):
    def __init__(self, n_feats, n_class, fc_hidden=512, *args, **kwargs):
        super().__init__(n_feats, n_class, *args, **kwargs)
        self.rnn = nn.LSTM(input_size=n_feats, num_layers=3, hidden_size=fc_hidden, batch_first=True)
        self.fc_hidden = fc_hidden
        self.n_feats = n_feats
        self.clf = nn.Linear(fc_hidden, n_class)

    def forward(self, spectrogram, spectrogram_length, *args, **kwargs):
        mel_spectrogram = spectrogram.clamp(min=1e-5).log()
        padded_input = nn.utils.rnn.pack_padded_sequence(
            mel_spectrogram, spectrogram_length, batch_first=True, enforce_sorted=False)
        packed_output, _ = self.rnn(padded_input)
        output, input_sizes = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        logits = self.clf(output)
        return {"logits": logits}

    def transform_input_lengths(self, input_lengths):
        return input_lengths  # we don't reduce time dimension here
