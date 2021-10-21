import torch
import torchaudio
from torch import Tensor

from hw_asr.augmentations.base import AugmentationBase


class FrequencyMasking(AugmentationBase):
    def __init__(self, *args, **kwargs):
        self._aug = torchaudio.transforms.FrequencyMasking(*args, **kwargs)

    def __call__(self, data: Tensor):
        x = data.unsqueeze(1).transpose(1, 2)    # freq-first
        return self._aug(x).squeeze(1).transpose(1, 2)    # time-first
