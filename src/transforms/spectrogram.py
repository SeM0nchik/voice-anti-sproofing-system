import torch
import torchaudio
from torch import nn


class LogPowerSpectrogram(nn.Module):
    def __init__(self, n_fft, hop_length, power, eps):
        super().__init__()

        self.eps = eps
        self.spectrogram = torchaudio.transforms.Spectrogram(
            n_fft=n_fft,
            hop_length=hop_length,
            window_fn=torch.blackman_window,
            power=power,
        )

    def forward(self, x):
        spec = self.spectrogram(x)
        return torch.log(spec + self.eps)
