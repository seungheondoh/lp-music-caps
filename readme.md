# :sound: LP-MusicCaps: LLM-Based Pseudo Music Captioning

[![Demo Video](https://i.imgur.com/cgi8NsD.jpg)](https://youtu.be/ezwYVaiC-AM)

This is a implementation of [LP-MusicCaps: LLM-Based Pseudo Music Captioning](https://arxiv.org/abs/2307.16372). This project aims to generate captions for music. 1) Tag-to-Caption: Using existing tags, We leverage the power of OpenAI's GPT-3.5 Turbo API to generate high-quality and contextually relevant captions based on music tag. 2) Audio-to-Caption: Using music-audio and pseudo caption pairs, we train a cross-model encoder-decoder model for end-to-end music captioning

> [**LP-MusicCaps: LLM-Based Pseudo Music Captioning**](https://arxiv.org/abs/2307.16372)

> SeungHeon Doh, Keunwoo Choi, Jongpil Lee, Juhan Nam   
> To appear ISMIR 2023   


## News

- **23.12.12** Our paper has been invited to the TISMIR journal. Stay tuned for the extended version
- **23.11.10** Our paper has been nominated for the ISMIR Best Paper Award (5/104)

## TL;DR


<p align = "center">
<img src = "https://i.imgur.com/2LC0nT1.png">
</p>

- Step 1.**[Tag-to-Caption: LLM Captioning](https://github.com/seungheondoh/lp-music-caps/tree/main/lpmc/llm_captioning)**: Generate caption from given tag input.
- Step 2.**[Pretrain Music Captioning Model](https://github.com/seungheondoh/lp-music-caps/tree/main/lpmc/music_captioning)**: Generate pseudo caption from given audio.
- Step 3.**[Transfer Music Captioning Model](https://github.com/seungheondoh/lp-music-caps/tree/main/lpmc/music_captioning/transfer.py)**: Generate human level caption from given audio.

## Open Source Material

- [Pre-trained model & Transfer model](https://huggingface.co/seungheondoh/lp-music-caps) 
- [Music & pseudo-caption dataset](https://huggingface.co/datasets/seungheondoh/LP-MusicCaps-MSD)
- [Huggingface demo](https://huggingface.co/spaces/seungheondoh/LP-Music-Caps-demo) 

are available online for future research. example of dataset in [notebook](https://github.com/seungheondoh/lp-music-caps/blob/main/notebook/Dataset.ipynb)


## Installation
To run this project locally, follow the steps below:

1. Install python and PyTorch:
    - python==3.10
    - torch==1.13.1 (Please install it according to your [CUDA version](https://pytorch.org/get-started/previous-versions/).)
    
2. Other requirements:
    - pip install -e .


## Quick Start: Tag to Caption

```bash
cd lpmc/llm_captioning
python run.py --prompt {writing, summary, paraphrase, attribute_prediction} --tags <music_tags>
```

Replace <music_tags> with the tags you want to generate captions for. Separate multiple tags with commas, such as `beatbox, finger snipping, male voice, amateur recording, medium tempo`.

tag_to_caption generation `writing` results:
```
query: 
write a song description sentence including the following attributes
beatbox, finger snipping, male voice, amateur recording, medium tempo
----------
results: 
"Experience the raw and authentic energy of an amateur recording as mesmerizing beatbox rhythms intertwine with catchy finger snipping, while a soulful male voice delivers heartfelt lyrics on a medium tempo track."
```


## Quick Start: Audio to Caption

```bash
cd demo
python app.py

# or
cd lpmc/music_captioning
wget https://huggingface.co/seungheondoh/lp-music-caps/resolve/main/transfer.pth -O exp/transfer/lp_music_caps
python captioning.py --audio_path ../../dataset/samples/orchestra.wav
```

```
{'text': "This is a symphonic orchestra playing  a piece that's riveting, thrilling and exciting. 
The peace would be suitable in a movie when something grand and impressive happens. 
There are clarinets, tubas, trumpets and french horns being played. The brass instruments help create that sense of a momentous occasion.", 
'time': '0:00-10:00'}

{'text': 'This is a classical music piece from a movie soundtrack. 
There is a clarinet playing the main melody while a brass section and a flute are playing the melody. 
The rhythmic background is provided by the acoustic drums. The atmosphere is epic and victorious. 
This piece could be used in the soundtrack of a historical drama movie during the scenes of an army marching towards the end.', 
'time': '10:00-20:00'}

{'text': 'This is a live performance of a classical music piece. There is a harp playing the melody while a horn is playing the bass line in the background. 
The atmosphere is epic. This piece could be used in the soundtrack of a historical drama movie during the scenes of an adventure video game.', 
'time': '20:00-30:00'}
```

## Re-Implementation
Checking `lpmc/llm_captioning` and `lpmc/music_captioning`

### License
This project is under the CC-BY-NC 4.0 license. See LICENSE for details.

### Acknowledgement
We would like to thank the [WavCaps](https://github.com/XinhaoMei/WavCaps) for audio-captioning training code and [deezer-playntell](https://github.com/deezer/playntell) for contents based captioning evaluation protocol. We would like to thank OpenAI for providing the GPT-3.5 Turbo API, which powers this project.

### Citation
Please consider citing our paper in your publications if the project helps your research. BibTeX reference is as follow.

```
@article{doh2023lp,
  title={LP-MusicCaps: LLM-Based Pseudo Music Captioning},
  author={Doh, SeungHeon and Choi, Keunwoo and Lee, Jongpil and Nam, Juhan},
  journal={arXiv preprint arXiv:2307.16372},
  year={2023}
}
```
