import torch_audiomentations
from torch import Tensor

from hw_asr.augmentations.base import AugmentationBase


class AddColoredNoise(AugmentationBase):
    """
        p -- probability of augmentation application
        min_f_decay, max_f_decay -- min and max values for f_decay
            (f_decay = 0 -- white noise)
        sr -- sample rate
    """
    def __init__(self, *args, **kwargs):
        self._aug = torch_audiomentations.AddColoredNoise(*args, **kwargs)

    def __call__(self, data: Tensor):
        x = data.unsqueeze(1)
        return self._aug(x).squeeze(1)
