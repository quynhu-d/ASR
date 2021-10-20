import torch_audiomentations
from torch import Tensor

from hw_asr.augmentations.base import AugmentationBase

# TODO: get noise folder path and pass args (update config too)
# NOISE_FOLDER_PATH = ""

class AddBackgroundNoise(AugmentationBase):
    """
        :param background_paths: Either a path to a folder with audio files or a list of paths
            to audio files.
        :param min_snr_in_db: minimum SNR in dB.
        :param max_snr_in_db: maximium SNR in dB.
        :param mode:
        :param p:
        :param p_mode:
        :param sample_rate:
    """
    def __init__(self, *args, **kwargs):
        self._aug = torch_audiomentations.AddBackgroundNoise(*args, **kwargs)

    def __call__(self, data: Tensor):
        x = data.unsqueeze(1)
        return self._aug(x).squeeze(1)
