import torch_audiomentations
from torch import Tensor

from hw_asr.augmentations.base import AugmentationBase

# TODO: add path to rir audio and update config


class ApplyImpulseResponse(AugmentationBase):
    """
        :param ir_paths: Either a path to a folder with audio files or a list of paths to audio files.
        :param convolve_mode:
        :param compensate_for_propagation_delay: Convolving audio with a RIR normally
            introduces a bit of delay, especially when the peak absolute amplitude in the
            RIR is not in the very beginning. When compensate_for_propagation_delay is
            set to True, the returned slices of audio will be offset to compensate for
            this delay.
        :param mode:
        :param p:
        :param p_mode:
        :param sample_rate:
    """

    def __init__(self, *args, **kwargs):
        self._aug = torch_audiomentations.ApplyImpulseResponse(*args, **kwargs)

    def __call__(self, data: Tensor):
        x = data.unsqueeze(1)
        return self._aug(x).squeeze(1)
