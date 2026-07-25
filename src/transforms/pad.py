from torch import nn


class ReplicationPad(nn.Module):
    def __init__(self, target_seconds=5.0, sample_rate=16000):
        super().__init__()

        self.target_len = int(target_seconds * sample_rate)

    def forward(self, x):
        num_samples = x.shape[-1]

        if num_samples >= self.target_len:
            return x[..., : self.target_len]

        num_repeats = self.target_len // num_samples + 1
        x = x.repeat(1, num_repeats)
        return x[..., : self.target_len]
