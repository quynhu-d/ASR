# ASR Project

## Installation

```shell
pip install -r ./requirements.txt
```
## Report
Report at [colab notebook](https://colab.research.google.com/drive/13EARXqEvQ7H13YszFym4_AKno0Q6MeQL?usp=sharing) includes:
- running of tests
- examples of augmentations
- how to run train.py and test.py
- which models were used with which configs
- results of the runs (links to [wandb](https://wandb.ai/quynhu_d/asr_project/table?workspace=user-quynhu_d) logs)
- calculating cer and wer for predictions on test-clean

## Model
Best model is located at `./saved_deepspeech/`

## Train
```
python ./train.py -c ./hw_asr/config/deepspeech_config.json
```
(Check `data_dir` in data_parameters depending on where you run train.py)

## Test
```
python ./test.py -r ./saved_deepspeech/model_best.pth -o deepspeech2_test_output.json
```
(Currently `config.json` file in `saved_deepspeech` has parameter `beam_search` set as false, change to true to use beam search)
## Credits

this repository is based on a heavily modified fork
of [pytorch-template](https://github.com/victoresque/pytorch-template) repository.
