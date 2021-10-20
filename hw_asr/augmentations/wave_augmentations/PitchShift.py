import torch_audiomentations
from torch import Tensor

from hw_asr.augmentations.base import AugmentationBase


class PitchShift(AugmentationBase):
    """
        :param sample_rate:
        :param min_transpose_semitones: Minimum pitch shift transposition in semitones (default -4.0)
        :param max_transpose_semitones: Maximum pitch shift transposition in semitones (default +4.0)
        :param mode: ``per_example``, ``per_channel``, or ``per_batch``. Default ``per_example``.
        :param p:
        :param p_mode:

    """
    def __init__(self, *args, **kwargs):
        self._aug = torch_audiomentations.PitchShift(*args, **kwargs)

    def __call__(self, data: Tensor):
        x = data.unsqueeze(1)
        return self._aug(x).squeeze(1)
