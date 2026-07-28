# Voice Anti-Spoofing System (ASVspoof2019 LA, LCNN)

<p align="center">
  <a href="#about">About</a> •
  <a href="#installation">Installation</a> •
  <a href="#dataset">Dataset</a> •
  <a href="#how-to-use">How To Use</a> •
  <a href="#results">Results</a> •
  <a href="#project-structure">Project Structure</a> •
  <a href="#credits--citation">Credits</a> •
  <a href="#license">License</a>
</p>

## About

This repository implements a **Countermeasure (CM) system for voice anti-spoofing** on the
**Logical Access (LA)** partition of the [ASVspoof2019](https://www.asvspoof.org/index2019.html)
dataset, using the **LCNN (Light CNN)** architecture, implemented from scratch following the
[STC Antispoofing System paper](https://arxiv.org/abs/1904.05576).

The model receives an STFT-based spectrogram of an utterance and predicts whether it is
`bonafide` (genuine human speech) or `spoof` (synthesized/converted/replayed speech).

This project was built for the HSE Deep Learning in Audio (DLA) course homework, on top of
[Blinorot/pytorch_project_template](https://github.com/Blinorot/pytorch_project_template).

## Installation

Installation may depend on your task. The general steps are the following:

0. (Optional) Create and activate new environment using [`conda`](https://conda.io/projects/conda/en/latest/user-guide/getting-started.html) or `venv` ([`+pyenv`](https://github.com/pyenv/pyenv)).

   a. `conda` version:

   ```bash
   # create env
   conda create -n project_env python=PYTHON_VERSION

   # activate env
   conda activate project_env
   ```

   b. `venv` (`+pyenv`) version:

   ```bash
   # create env
   ~/.pyenv/versions/PYTHON_VERSION/bin/python3 -m venv project_env

   # alternatively, using default python version
   python3 -m venv project_env

   # activate env
   source project_env/bin/activate
   ```

1. Install all required packages

   ```bash
   pip install -r requirements.txt
   ```

2. Install `pre-commit`:
   ```bash
   pre-commit install
   ```


## Dataset

The model is trained/evaluated on the **LA partition of ASVspoof2019**
(`ASVspoof2019_LA_{train,dev,eval}/flac/*.flac` + `ASVspoof2019.LA.cm.*.trn/trl.txt` protocols).

Dataset paths are configured in `src/configs/datasets/`:

- `la.yaml` — full `train`/`dev`/`eval` split, expects Kaggle-style paths
  (`/kaggle/input/awsaf49/asvpoof-2019-dataset/LA/LA`). Use this when running on Kaggle/Colab
  or once you've downloaded the full dataset locally with matching paths.

## How To Use

Here is a simple example how to train a model using a kaggle notebook:

### Clone repository from github

```bash
!git clone --depth 1 https://github.com/SeM0nchik/voice-anti-sproofing-system
%cd voice-anti-sproofing-system
```

### Install dependencies

```bash
!pip install -r requirements.txt
```
### Log in `wandb` or `conda` using api keys

```bash
import wandb
from kaggle_secrets import UserSecretsClient

user_secrets = UserSecretsClient()
wandb_key = user_secrets.get_secret("WANDB_KEY")

wandb.login(key=wandb_key)
```
Configs for logging provided in `src/configs/writer`.

### Train a model

```bash
python3 train.py -cn=baseline HYDRA_CONFIG_ARGUMENTS
```

Trains an LCNN model on `train`, evaluates on `dev`/`eval` every epoch, and saves checkpoints to `saved/<run_name>/`.

### Inference / evaluation

```bash
python3 inference.py inferencer.from_pretrained=PATH_TO_CHECKPOINT.pth
```

## Results

| Checkpoint (epoch) | EER (eval, %) |
|---|---|
| 10 | **5.276** |

EER is computed on the full ASVspoof2019 LA eval set (71 237 utterances) with
[this script](https://github.com/Blinorot/deep-learning-research/blob/summer_2026/hw/calculate_eer.py).

## Project Structure

```
.
├── src
│   ├── configs
│   │   ├── dataloader
│   │   ├── datasets
│   │   ├── metrics
│   │   ├── model
│   │   ├── transforms
│   │   │   ├── batch_transforms
│   │   │   └── instance_transforms
│   │   ├── writer
│   │   └── ...
│   ├── datasets
│   ├── logger
│   ├── loss
│   ├── metrics
│   ├── model
│   ├── trainer
│   ├── transforms
│   └── utils
├── inference.py
├── README.md
├── requirements.txt
├── train.py
└── ...
```

## Credits

This project is based on the [PyTorch Project Template](https://github.com/Blinorot/pytorch_project_template) by Petr Grinberg:

```
Grinberg, P. (2024). PyTorch Project Template [Computer software]. https://github.com/Blinorot/pytorch_project_template
```

The LCNN architecture follows [STC Antispoofing Systems for the ASVspoof2019 Challenge](https://arxiv.org/abs/1904.05576):

```
Lavrentyeva, G., Novoselov, S., Tseren, A., Volkova, M., Gorlanov, A., & Kozlov, A. (2019).
STC Antispoofing Systems for the ASVspoof2019 Challenge. arXiv:1904.05576.
```

## License

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)
