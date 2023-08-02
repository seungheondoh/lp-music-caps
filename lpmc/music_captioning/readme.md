# Audio-to-Caption Generation using Cross Modal Encoder-Decoder


<p align = "center">
<img src = "https://i.imgur.com/zsUmlcC.png">
</p>

## 1. Supervised Model

Download [MusicCaps audio](https://github.com/seungheondoh/music_caps_dl) or [MSD audio](https://github.com/SeungHeonDoh/msd-subsets) 

if you hard to get audio please request to me
- seungheondoh@kaist.ac.kr

```
# train supervised baseline model
python train.py --framework supervised --train_data mc --caption_type gt --warmup_epochs 1 --label_smoothing 0.1 --max_length 128 --batch-size 64 

# inference caption
python infer.py --framework supervised --train_data mc --caption_type gt --num_beams 5 --model_type last

# eval
python eval.py --framework supervised --caption_type gt
```


## 1. Supervised Model

Download [MusicCaps audio](https://github.com/seungheondoh/music_caps_dl), if you hard to get audio please request to me

- seungheondoh@kaist.ac.kr

```
# train supervised baseline model
python train.py --framework supervised --train_data mc --caption_type gt --warmup_epochs 1 --label_smoothing 0.1 --max_length 128 --batch-size 64 --epochs 100

# inference caption
python infer.py --framework supervised --train_data mc --caption_type gt --num_beams 5 --model_type last

# eval
python eval.py --framework supervised --caption_type gt
```

## 2. Pretrain, Transfer Learning

Download MSD audio, if you hard to get audio please request to me
- seungheondoh@kaist.ac.kr

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
