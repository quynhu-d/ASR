import unittest
import torch

from hw_asr.text_encoder.ctc_char_text_encoder import CTCCharTextEncoder


class TestTextEncoder(unittest.TestCase):
    def test_ctc_decode(self):
        text_encoder = CTCCharTextEncoder.get_simple_alphabet()
        text = "i^^ ^w^i^sss^hhh^   i ^^^s^t^aaaar^teee^d dddddd^oooo^in^g tttttttth^iiiis h^^^^^^^^w^ e^a^r^li^er"
        true_text = "i wish i started doing this hw earlier"
        inds = [text_encoder.char2ind[c] for c in text]
        decoded_text = text_encoder.ctc_decode(inds)
        self.assertIn(decoded_text, true_text)

    def test_beam_search(self):
        # TODO: (optional) write tests for beam search
        text_encoder = CTCCharTextEncoder('ab')
        probs = torch.tensor([[.2, .5, .3], [.8, .1, .1], [.1, .85, .05]])
        true = [('aa', .3400000035762787), ('ba', .20400001108646393), ('a', .13600000739097595)]
        hypos = text_encoder.ctc_beam_search(probs, 3)
        self.assertListEqual(true, hypos)
