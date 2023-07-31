# :sound: LP-MusicCaps: LLM-Based Pseudo Music Captioning

[![Demo Video](https://i.imgur.com/cgi8NsD.jpg)](https://youtu.be/ezwYVaiC-AM)

This is a implementation of [LP-MusicCaps: LLM-Based Pseudo Music Captioning](#). This project aims to generate captions for music. 1) Tag-to-Caption: Using existing tags, We leverage the power of OpenAI's GPT-3.5 Turbo API to generate high-quality and contextually relevant captions based on music tag. 2) Audio-to-Caption: Using music-audio and pseudo caption pairs, we train a cross-model encoder-decoder model for end-to-end music captioning

> [**LP-MusicCaps: LLM-Based Pseudo Music Captioning**](#)   
> SeungHeon Doh, Keunwoo Choi, Jongpil Lee, Juhan Nam   
> To appear ISMIR 2023   


## TL;DR


<p align = "center">
<img src = "https://i.imgur.com/2LC0nT1.png">
</p>

- **[1.Tag-to-Caption: LLM Captioning](https://github.com/seungheondoh/lp-music-caps/tree/main/lpmc/llm_captioning)**: Generate caption from given tag input.
- **[2.Audio-to-Caption: Pretrain Captioning Model](https://github.com/seungheondoh/lp-music-caps/tree/main/lpmc/music_captioning)**: Generate pseudo caption from given audio.
- **[3.Audio-to-Caption: Transfer Captioning Model](https://github.com/seungheondoh/lp-music-caps/tree/main/lpmc/music_captioning/transfer.py)**: Generate human level caption from given audio.

## Open Source Material

- [pre-trained models](https://huggingface.co/seungheondoh/lp-music-caps) 
- [music-pseudo caption dataset](https://huggingface.co/datasets/seungheondoh/LP-MusicCaps-MSD)
- [demo](https://huggingface.co/spaces/seungheondoh/LP-Music-Caps-demo) 

are available online for future research.


## Installation
To run this project locally, follow the steps below:

```
pip install -r requirements.txt
pip install -e .
```

## Quick Start: Tag to Caption

```bash
cd lmpc/llm_captioning
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
cd lmpc/music_captioning
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
We would like to thank the [WavCaps](https://github.com/XinhaoMei/WavCaps) for audio-captioning training code and [deezer-playntell](https://github.com/deezer/playntell) for contents based captioning evaluation protocal.

### Citation
Please consider citing our paper in your publications if the project helps your research. BibTeX reference is as follow.

```
Update soon
```