import torch
import torchaudio
from torch import Tensor

from hw_asr.augmentations.base import AugmentationBase


class TimeStretch(AugmentationBase):
    def __init__(self, *args, **kwargs):
        self._aug = torchaudio.transforms.TimeStretch(*args, **kwargs)

    def __call__(self, data: Tensor):
        x = data.type(torch.complex128)    # pass complex spectrogram of shape B * F * T
        return self._aug(x).type(torch.float64)
