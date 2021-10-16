import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Sequential

from hw_asr.base import BaseModel


class ActDropNormCNN1D(nn.Module):
    def __init__(self, n_feats, dropout, keep_shape=False):
        super(ActDropNormCNN1D, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(n_feats)
        self.keep_shape = keep_shape

    def forward(self, x):
        x = x.transpose(1, 2)
        # x = self.norm(self.dropout(F.gelu(x)))
        x = self.dropout(F.gelu(self.norm(x)))
        if self.keep_shape:
            return x.transpose(1, 2)
        else:
            return x


class SpeechRecognition(BaseModel):
    hyper_parameters = {
        "num_classes": 29,
        "n_feats": 81,
        "dropout": 0.1,
        "hidden_size": 1024,
        "num_layers": 1
    }

    def __init__(self, n_feats, n_class, fc_hidden=512, *args, **kwargs):
        super(SpeechRecognition, self).__init__()
        self.num_layers = self.hyper_parameters['num_layers']
        self.fc_hidden = fc_hidden
        self.cnn = nn.Sequential(
            nn.Conv1d(n_feats, n_feats, 10, 2, padding=10 // 2),
            ActDropNormCNN1D(n_feats, .2),
        )
        self.dense = nn.Sequential(
            nn.Linear(n_feats, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(.2),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(.2),
        )
        self.lstm = nn.LSTM(input_size=128, hidden_size=fc_hidden,
                            num_layers=self.num_layers, dropout=0.0,
                            bidirectional=False)
        self.layer_norm2 = nn.LayerNorm(fc_hidden)
        self.dropout2 = nn.Dropout(.2)
        self.final_fc = nn.Linear(fc_hidden, n_class)

    def _init_hidden(self, batch_size):
        n, hs = self.num_layers, self.hidden_size
        return (torch.zeros(n * 1, batch_size, hs),
                torch.zeros(n * 1, batch_size, hs))

    def forward(self, spectrogram, hidden):
        x = spectrogram
        x = x.squeeze(1)  # batch, feature, time
        x = self.cnn(x)  # batch, time, feature
        x = self.dense(x)  # batch, time, feature
        x = x.transpose(0, 1)  # time, batch, feature
        out, (hn, cn) = self.lstm(x, hidden)
        x = self.dropout2(F.gelu(self.layer_norm2(out)))  # (time, batch, n_class)
        return self.final_fc(x), (hn, cn)

class CNNModel(BaseModel):
    def __init__(self, n_feats, n_class, fc_hidden=512, *args, **kwargs):
        super().__init__(n_feats, n_class, *args, **kwargs)
        self.net = Sequential(
            # people say it can aproximate any function...
            nn.Linear(in_features=n_feats, out_features=fc_hidden),
            nn.ReLU(),
            nn.Linear(in_features=fc_hidden, out_features=fc_hidden),
            nn.ReLU(),
            nn.Linear(in_features=fc_hidden, out_features=n_class)
        )

    def forward(self, spectrogram, *args, **kwargs):
        return {"logits": self.net(spectrogram)}

    def transform_input_lengths(self, input_lengths):
        return input_lengths  # we don't reduce time dimension here
