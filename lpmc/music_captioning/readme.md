# Audio-to-Caption using Cross Modal Encoder-Decoder

We used a cross-modal encoder-decoder transformer architecture. 

1. Similar to Whisper, the encoder takes a log-mel spectrogram with six convolution layers with a filter width of 3 and the GELU activation function. With the exception of the first layer, each convolution layer has a stride of two. The output of the convolution layers is combined with the sinusoidal position encoding and then processed by the encoder transformer blocks. 

2. Following the BART architecture, our encoder and decoder both have 768 widths and 6 transformer blocks. The decoder processes tokenized text captions using transformer blocks with a multi-head attention module that includes a mask to hide future tokens for causality. The music and caption representations are fed into the cross-modal attention layer, and the head of the language model in the decoder predicts the next token autoregressively using the cross-entropy loss.

- **Supervised Model** : [download link](https://huggingface.co/seungheondoh/lp-music-caps/resolve/main/supervised.pth)
- **Pretrain Model** : [download link](https://huggingface.co/seungheondoh/lp-music-caps/resolve/main/pretrain.pth)
- **Transfer Model** : [download link](https://huggingface.co/seungheondoh/lp-music-caps/resolve/main/transfer.pth)

<p align = "center">
  <img src = "https://i.imgur.com/zsUmlcC.png" width="600">
</p>

## 0. Quick Start
```bash
# download pretrain model weight from huggingface

wget https://huggingface.co/seungheondoh/lp-music-caps/resolve/main/supervised.pth -O exp/supervised/gt/last.pth
wget https://huggingface.co/seungheondoh/lp-music-caps/resolve/main/transfer.pth -O exp/transfer/lp_music_caps/last.pth
wget https://huggingface.co/seungheondoh/lp-music-caps/resolve/main/pretrain.pth -O exp/pretrain/lp_music_caps/last.pth
python captioning.py --audio_path ../../dataset/samples/orchestra.wav
```

```bash
{
  'text': "This is a symphonic orchestra playing  a piece that's riveting, thrilling and exciting. 
          The peace would be suitable in a movie when something grand and impressive happens. 
          There are clarinets, tubas, trumpets and french horns being played. The brass instruments help create that sense of a momentous occasion.", 
  'time': '0:00-10:00'
} 
{
  'text': 'This is a classical music piece from a movie soundtrack. 
          There is a clarinet playing the main melody while a brass section and a flute are playing the melody. 
          The rhythmic background is provided by the acoustic drums. The atmosphere is epic and victorious. 
          This piece could be used in the soundtrack of a historical drama movie during the scenes of an army marching towards the end.', 
'time': '10:00-20:00'
}
```

## 1. Preprocessing audio with ffmpeg

For fast training, we resample audio at 16000 sampling rate and save it as `.npy`.

```python
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
```

The code using `multiprocessing`` is as follows. We also provide preprocessing code for balanced data loading.

```
# multiprocessing resampling & bulid tag-to-track linked list for balanced data loading.
python preprocessor.py
```

## 2. Train & Eval Supervised Model (Baseline)

Download [MusicCaps audio](https://github.com/seungheondoh/music_caps_dl), if you hard to get audio please request for research purpose

```
# train supervised baseline model
python train.py --framework supervised --train_data mc --caption_type gt --warmup_epochs 1 --label_smoothing 0.1 --max_length 128 --batch-size 64 --epochs 100

# inference caption
python infer.py --framework supervised --train_data mc --caption_type gt --num_beams 5 --model_type last

# eval
python eval.py --framework supervised --caption_type gt
```

## 3. Pretrain, Transfer Music Captioning Model (Proposed)

Download MSD audio, if you hard to get audio please request for research purpose

```
# train pretrain model
python train.py --framework pretrain --train_data msd --caption_type lp_music_caps --warmup_epochs 125 --label_smoothing 0.1 --max_length 110 --batch-size 256 --epochs 4096

# train transfer model
python transfer.py --caption_type gt --warmup_epochs 1 --label_smoothing 0.1 --max_length 128 --batch-size 64 --epochs 100

# inference caption
python infer.py --framework transfer --caption_type lp_music_caps --num_beams 5 --model_type last

# eval
python eval.py --framework transfer --caption_type lp_music_caps
```

### License
This project is under the CC-BY-NC 4.0 license. See LICENSE for details.


### Acknowledgement
We would like to thank the [Whisper](https://github.com/openai/whisper) for audio frontend, [WavCaps](https://github.com/XinhaoMei/WavCaps) for audio-captioning training code and [deezer-playntell](https://github.com/deezer/playntell) for contents based captioning evaluation protocal.