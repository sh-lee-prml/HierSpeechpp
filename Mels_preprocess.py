import numpy as np

np.random.seed(1234)

import torch
from torchaudio.transforms import MelSpectrogram, Spectrogram, MelScale

class MelSpectrogramFixed(torch.nn.Module):
    """In order to remove padding of torchaudio package + add log scale."""

    def __init__(self, **kwargs):
        super(MelSpectrogramFixed, self).__init__()
        self.torchaudio_backend = MelSpectrogram(**kwargs)

    def forward(self, x):
        outputs = torch.log(self.torchaudio_backend(x) + 0.001)

        return outputs[..., :-1]

class SpectrogramFixed(torch.nn.Module):
    """In order to remove padding of torchaudio package + add log10 scale."""

    def __init__(self, **kwargs):
        super(SpectrogramFixed, self).__init__()
        self.torchaudio_backend = Spectrogram(**kwargs)

    def forward(self, x):
        outputs = self.torchaudio_backend(x)

        return outputs[..., :-1]

class MelfilterFixed(torch.nn.Module):
    """In order to remove padding of torchaudio package + add log10 scale."""

    def __init__(self, **kwargs):
        super(MelfilterFixed, self).__init__()
        self.torchaudio_backend = MelScale(**kwargs)

    def forward(self, x):
        outputs = torch.log(self.torchaudio_backend(x) + 0.001)

        return outputs