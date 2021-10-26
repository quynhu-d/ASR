from typing import List, Union
import youtokentome as yttm
import numpy as np
from torch import Tensor
from pathlib import Path
from hw_asr.utils import ROOT_PATH

from hw_asr.text_encoder.ctc_char_text_encoder import CTCCharTextEncoder


class BPETextEncoder(CTCCharTextEncoder):

    def __init__(self, model_path: str = None, train_data: str = None, vocab_size: int = 5000):
        assert (model_path is not None) or (train_data is not None)
        if train_data:
            print('Training BPE encoder')
            if model_path is None:
                model_path = ROOT_PATH / "hw_asr" / "bpe_saved"
                model_path.mkdir(exist_ok=True, parents=True)
                model_path = str(Path(model_path) / "bpe.model")
            self.bpe = yttm.BPE.train(train_data, vocab_size=vocab_size, model=model_path)
        else:
            self.bpe = yttm.BPE(model_path)
        super().__init__(self.bpe.vocab())

    def encode(self, text) -> Tensor:
        text = self.normalize_text(text)
        try:
            return Tensor(self.bpe.encode(text)).unsqueeze(0)
        except KeyError as e:
            raise Exception(f"Can't encode text '{text}'.")

    def decode(self, vector: Union[Tensor, np.ndarray, List[int]]):
        if len(vector) == 0:
            return ""
        return self.bpe.decode(vector.tolist())[0].strip()

    def ctc_decode(self, inds: List[int]) -> str:
        res = []
        last_blank = False
        if isinstance(inds, Tensor):
            inds = inds.squeeze().tolist()
        for ind in inds:
            if ind == self.char2ind[self.EMPTY_TOK]:
                last_blank = True
            else:
                if len(res) == 0 or last_blank or res[-1] != ind:
                    res.append(ind)
                last_blank = False
        if len(res) == 0:
            return ""
        return self.decode(res)[0]
