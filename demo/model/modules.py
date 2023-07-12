### code reference: https://github.com/openai/whisper/blob/main/whisper/audio.py

import os
import torch
import torchaudio
import numpy as np
import torch.nn.functional as F
from torch import Tensor, nn
from typing import Dict, Iterable, Optional

# hard-coded audio hyperparameters
SAMPLE_RATE = 16000
N_FFT = 1024
N_MELS = 128
HOP_LENGTH = int(0.01 * SAMPLE_RATE)
DURATION = 10
N_SAMPLES = int(DURATION * SAMPLE_RATE) 
N_FRAMES = N_SAMPLES // HOP_LENGTH + 1 

def sinusoids(length, channels, max_timescale=10000):
    """Returns sinusoids for positional embedding"""
    log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
    inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2))
    scaled_time = torch.arange(length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
    return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)

class MelEncoder(nn.Module):
    """
    time-frequency represntation
    """
    def __init__(self, 
                sample_rate= 16000,
                f_min=0,
                f_max=8000,
                n_fft=1024,
                win_length=1024,
                hop_length = int(0.01 * 16000),
                n_mels = 128,
                power = None,
                pad= 0,
                normalized= False,
                center= True,
                pad_mode= "reflect"
                ):
        super(MelEncoder, self).__init__()
        self.window = torch.hann_window(win_length)
        self.spec_fn = torchaudio.transforms.Spectrogram(
            n_fft = n_fft,
            win_length = win_length,
            hop_length = hop_length,
            power = power
        )
        self.mel_scale = torchaudio.transforms.MelScale(
            n_mels, 
            sample_rate,
            f_min,
            f_max,
            n_fft // 2 + 1)
        
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()

    def forward(self, wav):
        spec = self.spec_fn(wav)
        power_spec = spec.real.abs().pow(2)
        mel_spec = self.mel_scale(power_spec)
        mel_spec = self.amplitude_to_db(mel_spec) # Log10(max(reference value and amin))
        return mel_spec

class AudioEncoder(nn.Module):
    def __init__(
        self, n_mels: int, n_ctx: int, audio_dim: int, text_dim: int, num_of_stride_conv: int,
    ):
        super().__init__()
        self.mel_encoder = MelEncoder(n_mels=n_mels)
        self.conv1 = nn.Conv1d(n_mels, audio_dim, kernel_size=3, padding=1)
        self.conv_stack = nn.ModuleList([])
        for _ in range(num_of_stride_conv):
            self.conv_stack.append(
                nn.Conv1d(audio_dim, audio_dim, kernel_size=3, stride=2, padding=1)
            )
        # self.proj = nn.Linear(audio_dim, text_dim, bias=False)
        self.register_buffer("positional_embedding", sinusoids(n_ctx, text_dim))

    def forward(self, x: Tensor):
        """
        x : torch.Tensor, shape = (batch_size, waveform)
            single channel wavform
        """
        x = self.mel_encoder(x) # (batch_size, n_mels, n_ctx)
        x = F.gelu(self.conv1(x))
        for conv in self.conv_stack:
            x = F.gelu(conv(x))
        x = x.permute(0, 2, 1)
        x = (x + self.positional_embedding).to(x.dtype)
        return x