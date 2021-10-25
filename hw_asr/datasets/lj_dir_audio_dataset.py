import logging
from pathlib import Path
import pandas as pd
from hw_asr.datasets.custom_audio_dataset import CustomAudioDataset
from hw_asr.base.base_text_encoder import BaseTextEncoder
# from hw_asr.text_encoder.ctc_char_text_encoder import CTCCharTextEncoder
# from hw_asr.utils.parse_config import ConfigParser

logger = logging.getLogger(__name__)


class LJDirAudioDataset(CustomAudioDataset):
    def __init__(self, audio_dir, transcription_csv=None, *args, **kwargs):
        data = []
        transc_csv = pd.read_csv(transcription_csv, delimiter='|', header=None)
        transc_csv.rename(columns={0: 'id', 1: 'text', 2: 'norm_text'})
        for path in Path(audio_dir).iterdir():
            entry = {}
            if path.suffix in [".mp3", ".wav", ".flac", ".m4a"]:
                entry["path"] = str(path)
                name = path.stem
                entry["text"] = BaseTextEncoder.normalize_text(transc_csv[transc_csv.id == name].norm_text.iloc[0])
            if len(entry) > 0:
                data.append(entry)
        super().__init__(data, *args, **kwargs)

#
#
# if __name__ == "__main__":
#     text_encoder = CTCCharTextEncoder.get_simple_alphabet()
#     config_parser = ConfigParser.get_default_configs()
#
#     ds = LJDirAudioDataset("data/datasets/custom/audio", text_encoder=text_encoder, config_parser=config_parser)
#     item = ds[0]
#     print(item)
