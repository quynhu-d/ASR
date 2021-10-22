import torch
import torchaudio
from torch import Tensor
import numpy as np

from hw_asr.augmentations.base import AugmentationBase


class TimeStretch(AugmentationBase):
    def __init__(self, min_rate, max_rate, *args, **kwargs):
        self.min_rate = min_rate
        self.max_rate = max_rate
        self._aug = torchaudio.transforms.TimeStretch(*args, **kwargs)

    def __call__(self, data: Tensor):
        rate = np.random.uniform(self.min_rate, self.max_rate)    # choose random rate from given range
        x = data.type(torch.complex128)    # pass complex spectrogram of shape B * F * T
        return self._aug(x, rate).type(torch.float64)
