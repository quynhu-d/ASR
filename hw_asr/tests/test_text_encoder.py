import unittest
import torch
from hw_asr.utils import ROOT_PATH
from pathlib import Path
from hw_asr.text_encoder.ctc_char_text_encoder import CTCCharTextEncoder
from hw_asr.text_encoder.bpe_text_encoder import BPETextEncoder


class TestTextEncoder(unittest.TestCase):
    def test_ctc_decode(self):
        text_encoder = CTCCharTextEncoder.get_simple_alphabet()
        text = "i^^ ^w^i^sss^hhh^   i ^^^s^t^aaaar^teee^d dddddd^oooo^in^g tttttttth^iiiis h^^^^^^^^w^ e^a^r^li^er"
        true_text = "i wish i started doing this hw earlier"
        inds = [text_encoder.char2ind[c] for c in text]
        decoded_text = text_encoder.ctc_decode(inds)
        self.assertIn(decoded_text, true_text)

    def test_beam_search(self):
        text_encoder = CTCCharTextEncoder('a')
        probs = torch.tensor([[.3, .7], [.5, .5]])
        true = [('a', .5), ('a', .35)]    # a, a^
        hypos = text_encoder.ctc_beam_search(probs, probs_length=3, beam_size=2)
        self.assertListEqual([line for line, _ in true], [line for line, _ in hypos])
        for (_, el_true), (_, el_hypos) in zip(true, hypos):
            self.assertAlmostEqual(el_true, el_hypos)

    def test_bpe_encoder(self):
        # text_encoder = BPETextEncoder(train_data='../../data/datasets/librispeech/dev_clean_texts.txt')
        model_path = Path(ROOT_PATH / "hw_asr" / "bpe_saved" / "bpe.model")
        text_encoder = BPETextEncoder(model_path=str(model_path))
        print("BPE vocab size:", text_encoder.bpe.vocab_size())
        print("Encoder vocab size:", len(text_encoder))
        print("Some BPE subwords:", text_encoder.bpe.vocab()[:10])
        print(text_encoder.encode('i need some sleep'))
