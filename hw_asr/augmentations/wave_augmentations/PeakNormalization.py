import torch_audiomentations
from torch import Tensor

from hw_asr.augmentations.base import AugmentationBase


class PeakNormalization(AugmentationBase):
    """
    Apply a constant amount of gain, so that highest signal level present in each audio snippet
    in the batch becomes 0 dBFS, i.e. the loudest level allowed if all samples must be between
    -1 and 1.

    This transform has an alternative mode (apply_to="only_too_loud_sounds") where it only
    applies to audio snippets that have extreme values outside the [-1, 1] range. This is useful
    for avoiding digital clipping in audio that is too loud, while leaving other audio
    untouched.
    """

    def __init__(self, *args, **kwargs):
        self._aug = torch_audiomentations.PeakNormalization(*args, **kwargs)

    def __call__(self, data: Tensor):
        x = data.unsqueeze(1)
        return self._aug(x).squeeze(1)
