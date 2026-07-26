import torch
from torch import nn


class MaxFeatureMap(nn.Module):
  def __init__(self):
    super().__init__()

  def forward(self, x):
    x1, x2 = torch.chunk(x, 2, dim=1)
    return torch.maximum(x1, x2)


class ConvMFM(nn.Module):
  def __init__(self, in_chan, out_chan, kernel_size):
    super().__init__()

    self.conv = nn.Conv2d(
      in_channels=in_chan,
      out_channels=out_chan,
      kernel_size=kernel_size,
      stride=1,
      padding="same",
    )
    self.mfm = MaxFeatureMap()

  def forward(self, x):
    x = self.conv(x)
    x = self.mfm(x)
    return x


class LCNNModel(nn.Module):
  def __init__(self, n_freq, n_frames, dropout, n_class):
    super().__init__()

    self.features = nn.Sequential(
      ConvMFM(1, 64, kernel_size=5),
      nn.MaxPool2d(kernel_size=2, stride=2),

      ConvMFM(32, 64, kernel_size=1),
      nn.BatchNorm2d(32),
      ConvMFM(32, 96, kernel_size=3),
      nn.MaxPool2d(kernel_size=2, stride=2),
      nn.BatchNorm2d(48),

      ConvMFM(48, 96, kernel_size=1),
      nn.BatchNorm2d(48),
      ConvMFM(48, 128, kernel_size=3),
      nn.MaxPool2d(kernel_size=2, stride=2),

      ConvMFM(64, 128, kernel_size=1),
      nn.BatchNorm2d(64),
      ConvMFM(64, 64, kernel_size=3),
      nn.BatchNorm2d(32),
      ConvMFM(32, 64, kernel_size=1),
      nn.BatchNorm2d(32),
      ConvMFM(32, 64, kernel_size=3),
      nn.MaxPool2d(kernel_size=2, stride=2),
    )

    pooled_freq = n_freq // 16
    pooled_frames = n_frames // 16
    flatten_dim = 32 * pooled_freq * pooled_frames

    self.fc1 = nn.Linear(flatten_dim, 160)
    self.mfm_fc = MaxFeatureMap()
    self.dropout = nn.Dropout(p=dropout)
    self.batch_norm_fc = nn.BatchNorm1d(80)
    self.fc2 = nn.Linear(80, n_class)

    self.apply(self._init_weights)

  @staticmethod
  def _init_weights(module):
    if isinstance(module, (nn.Conv2d, nn.Linear)):
      nn.init.kaiming_normal_(module.weight)
      if module.bias is not None:
        nn.init.zeros_(module.bias)

  def forward(self, data_object, **batch):
    x = self.features(data_object)
    x = x.flatten(start_dim=1)
    x = self.fc1(x)
    x = self.mfm_fc(x)
    x = self.dropout(x)
    x = self.batch_norm_fc(x)
    logits = self.fc2(x)
    return {"logits": logits}
