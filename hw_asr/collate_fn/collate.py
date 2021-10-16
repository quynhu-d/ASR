import logging
from typing import List
import torch

logger = logging.getLogger(__name__)


def collate_fn(dataset_items: List[dict]):
    """
    Collate and pad fields in dataset items
    """

    bs = len(dataset_items)
    spec_ft_s = dataset_items[0]['spectrogram'].squeeze().shape[0]

    text_time_lengths = [item['text_encoded'].squeeze().shape[0] for item in dataset_items]
    spec_time_lengths = [item['spectrogram'].squeeze().shape[1] for item in dataset_items]
    batch_text_encoded = torch.zeros(bs, max(text_time_lengths))
    batch_spectrogram = torch.zeros(bs, max(spec_time_lengths), spec_ft_s)
    for i, (item, text_length, spec_length) in enumerate(zip(
            dataset_items, text_time_lengths, spec_time_lengths)):
        batch_text_encoded[i, :text_length] = item['text_encoded'].squeeze()
        batch_spectrogram[i, :spec_length, :] = item['spectrogram'].T.squeeze()
    return {
        'text_encoded': batch_text_encoded,
        'text_encoded_length': torch.tensor(text_time_lengths),
        'text': [item['text'] for item in dataset_items],
        'spectrogram': batch_spectrogram,
        'spectrogram_length': torch.tensor(spec_time_lengths),
        'audio': [item['audio'] for item in dataset_items],
        'duration': [item['duration'] for item in dataset_items]
    }
    # raise NotImplementedError
