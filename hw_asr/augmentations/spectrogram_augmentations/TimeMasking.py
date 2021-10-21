import torch
import torchaudio
from torch import Tensor

from hw_asr.augmentations.base import AugmentationBase


class TimeMasking(AugmentationBase):
    def __init__(self, *args, **kwargs):
        self._aug = torchaudio.transforms.TimeMasking(*args, **kwargs)

    def __call__(self, data: Tensor):
        x = data.transpose(1, 2).unsqueeze(1)    # freq-first
        print(x.shape)
        return self._aug(x).squeeze(1).transpose(1, 2)    # time-first
