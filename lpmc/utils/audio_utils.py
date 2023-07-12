STR_CLIP_ID = 'clip_id'
STR_AUDIO_SIGNAL = 'audio_signal'
STR_TARGET_VECTOR = 'target_vector'


STR_CH_FIRST = 'channels_first'
STR_CH_LAST = 'channels_last'

import io
import os
import tqdm
import logging
import subprocess
from typing import Tuple
from pathlib import Path

# import librosa
import numpy as np
import soundfile as sf

import itertools
from numpy.fft import irfft

def _resample_load_ffmpeg(path: str, sample_rate: int, downmix_to_mono: bool) -> Tuple[np.ndarray, int]:
    """
    Decoding, downmixing, and downsampling by librosa.
    Returns a channel-first audio signal.

    Args:
        path:
        sample_rate:
        downmix_to_mono:

    Returns:
        (audio signal, sample rate)
    """

    def _decode_resample_by_ffmpeg(filename, sr):
        """decode, downmix, and resample audio file"""
        channel_cmd = '-ac 1 ' if downmix_to_mono else ''  # downmixing option
        resampling_cmd = f'-ar {str(sr)}' if sr else ''  # downsampling option
        cmd = f"ffmpeg -i \"{filename}\" {channel_cmd} {resampling_cmd} -f wav -"
        p = subprocess.Popen(cmd, shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = p.communicate()
        return out

    src, sr = sf.read(io.BytesIO(_decode_resample_by_ffmpeg(path, sr=sample_rate)))
    return src.T, sr


def _resample_load_librosa(path: str, sample_rate: int, downmix_to_mono: bool, **kwargs) -> Tuple[np.ndarray, int]:
    """
    Decoding, downmixing, and downsampling by librosa.
    Returns a channel-first audio signal.
    """
    src, sr = librosa.load(path, sr=sample_rate, mono=downmix_to_mono, **kwargs)
    return src, sr


def load_audio(
    path: str or Path,
    ch_format: str,
    sample_rate: int = None,
    downmix_to_mono: bool = False,
    resample_by: str = 'ffmpeg',
    **kwargs,
) -> Tuple[np.ndarray, int]:
    """A wrapper of librosa.load that:
        - forces the returned audio to be 2-dim,
        - defaults to sr=None, and
        - defaults to downmix_to_mono=False.

    The audio decoding is done by `audioread` or `soundfile` package and ultimately, often by ffmpeg.
    The resampling is done by `librosa`'s child package `resampy`.

    Args:
        path: audio file path
        ch_format: one of 'channels_first' or 'channels_last'
        sample_rate: target sampling rate. if None, use the rate of the audio file
        downmix_to_mono:
        resample_by (str): 'librosa' or 'ffmpeg'. it decides backend for audio decoding and resampling.
        **kwargs: keyword args for librosa.load - offset, duration, dtype, res_type.

    Returns:
        (audio, sr) tuple
    """
    if ch_format not in (STR_CH_FIRST, STR_CH_LAST):
        raise ValueError(f'ch_format is wrong here -> {ch_format}')

    if os.stat(path).st_size > 8000:
        if resample_by == 'librosa':
            src, sr = _resample_load_librosa(path, sample_rate, downmix_to_mono, **kwargs)
        elif resample_by == 'ffmpeg':
            src, sr = _resample_load_ffmpeg(path, sample_rate, downmix_to_mono)
        else:
            raise NotImplementedError(f'resample_by: "{resample_by}" is not supposred yet')
    else:
        raise ValueError('Given audio is too short!')
    return src, sr

    # if src.ndim == 1:
    #     src = np.expand_dims(src, axis=0)
    # # now always 2d and channels_first

    # if ch_format == STR_CH_FIRST:
    #     return src, sr
    # else:
    #     return src.T, sr

def ms(x):
    """Mean value of signal `x` squared.
    :param x: Dynamic quantity.
    :returns: Mean squared of `x`.
    """
    return (np.abs(x)**2.0).mean()

def normalize(y, x=None):
    """normalize power in y to a (standard normal) white noise signal.
    Optionally normalize to power in signal `x`.
    #The mean power of a Gaussian with :math:`\\mu=0` and :math:`\\sigma=1` is 1.
    """
    if x is not None:
        x = ms(x)
    else:
        x = 1.0
    return y * np.sqrt(x / ms(y))

def noise(N, color='white', state=None):
    """Noise generator.
    :param N: Amount of samples.
    :param color: Color of noise.
    :param state: State of PRNG.
    :type state: :class:`np.random.RandomState`
    """
    try:
        return _noise_generators[color](N, state)
    except KeyError:
        raise ValueError("Incorrect color.")

def white(N, state=None):
    """
    White noise.
    :param N: Amount of samples.
    :param state: State of PRNG.
    :type state: :class:`np.random.RandomState`
    White noise has a constant power density. It's narrowband spectrum is therefore flat.
    The power in white noise will increase by a factor of two for each octave band,
    and therefore increases with 3 dB per octave.
    """
    state = np.random.RandomState() if state is None else state
    return state.randn(N)

def pink(N, state=None):
    """
    Pink noise.
    :param N: Amount of samples.
    :param state: State of PRNG.
    :type state: :class:`np.random.RandomState`
    Pink noise has equal power in bands that are proportionally wide.
    Power density decreases with 3 dB per octave.
    """
    state = np.random.RandomState() if state is None else state
    uneven = N % 2
    X = state.randn(N // 2 + 1 + uneven) + 1j * state.randn(N // 2 + 1 + uneven)
    S = np.sqrt(np.arange(len(X)) + 1.)  # +1 to avoid divide by zero
    y = (irfft(X / S)).real
    if uneven:
        y = y[:-1]
    return normalize(y)

def blue(N, state=None):
    """
    Blue noise.
    :param N: Amount of samples.
    :param state: State of PRNG.
    :type state: :class:`np.random.RandomState`
    Power increases with 6 dB per octave.
    Power density increases with 3 dB per octave.
    """
    state = np.random.RandomState() if state is None else state
    uneven = N % 2
    X = state.randn(N // 2 + 1 + uneven) + 1j * state.randn(N // 2 + 1 + uneven)
    S = np.sqrt(np.arange(len(X)))  # Filter
    y = (irfft(X * S)).real
    if uneven:
        y = y[:-1]
    return normalize(y)

def brown(N, state=None):
    """
    Violet noise.
    :param N: Amount of samples.
    :param state: State of PRNG.
    :type state: :class:`np.random.RandomState`
    Power decreases with -3 dB per octave.
    Power density decreases with 6 dB per octave.
    """
    state = np.random.RandomState() if state is None else state
    uneven = N % 2
    X = state.randn(N // 2 + 1 + uneven) + 1j * state.randn(N // 2 + 1 + uneven)
    S = (np.arange(len(X)) + 1)  # Filter
    y = (irfft(X / S)).real
    if uneven:
        y = y[:-1]
    return normalize(y)

def violet(N, state=None):
    """
    Violet noise. Power increases with 6 dB per octave.
    :param N: Amount of samples.
    :param state: State of PRNG.
    :type state: :class:`np.random.RandomState`
    Power increases with +9 dB per octave.
    Power density increases with +6 dB per octave.
    """
    state = np.random.RandomState() if state is None else state
    uneven = N % 2
    X = state.randn(N // 2 + 1 + uneven) + 1j * state.randn(N // 2 + 1 + uneven)
    S = (np.arange(len(X)))  # Filter
    y = (irfft(X * S)).real
    if uneven:
        y = y[:-1]
    return normalize(y)

_noise_generators = {
    'white': white,
    'pink': pink,
    'blue': blue,
    'brown': brown,
    'violet': violet,
}

def noise_generator(N=44100, color='white', state=None):
    """Noise generator.
    :param N: Amount of unique samples to generate.
    :param color: Color of noise.
    Generate `N` amount of unique samples and cycle over these samples.
    """
    #yield from itertools.cycle(noise(N, color)) # Python 3.3
    for sample in itertools.cycle(noise(N, color, state)):
        yield sample

def heaviside(N):
    """Heaviside.
    Returns the value 0 for `x < 0`, 1 for `x > 0`, and 1/2 for `x = 0`.
    """
    return 0.5 * (np.sign(N) + 1)