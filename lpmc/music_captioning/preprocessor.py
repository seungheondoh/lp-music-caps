import os
import random
from datasets import load_dataset
from contextlib import contextmanager
import multiprocessing
import numpy as np
import json
from lpmc.utils.audio_utils import load_audio, STR_CH_FIRST
from sklearn.preprocessing import MultiLabelBinarizer
from tqdm import tqdm

# hard coding hpamras
DATASET_PATH = "../../dataset/msd"
MUSIC_SAMPLE_RATE = 16000
DURATION = 30
DATA_LENGTH = int(MUSIC_SAMPLE_RATE * DURATION)

@contextmanager
def poolcontext(*args, **kwargs):
    pool = multiprocessing.Pool(*args, **kwargs)
    yield pool
    pool.terminate()
    
def msd_resampler(sample):
    path = sample['path']
    save_name = os.path.join(DATASET_PATH,'npy', path.replace(".mp3",".npy"))
    src, _ = load_audio(
        path=os.path.join(DATASET_PATH,'songs',path),
        ch_format= STR_CH_FIRST,
        sample_rate= MUSIC_SAMPLE_RATE,
        downmix_to_mono= True)
    if src.shape[-1] < DATA_LENGTH: # short case
        pad = np.zeros(DATA_LENGTH)
        pad[:src.shape[-1]] = src
        src = pad
    elif src.shape[-1] > DATA_LENGTH: # too long case
        src = src[:DATA_LENGTH]
    
    if not os.path.exists(os.path.dirname(save_name)):
        os.makedirs(os.path.dirname(save_name))
    np.save(save_name, src.astype(np.float32))
    
def build_tag_to_track(msd_dataset, split):
    """
    for balanced sampler, we bulid tag_to_track graph
    """
    mlb = MultiLabelBinarizer()
    indexs = [i['track_id'] for i in msd_dataset[split]]
    binary = mlb.fit_transform([i['tag'] for i in msd_dataset[split]])
    tags = list(mlb.classes_)
    tag_to_track = {}
    for idx, tag in enumerate(tqdm(tags)):
        track_list = [indexs[i] for i in binary[:,idx].nonzero()[0]]
        tag_to_track[tag] = track_list

    with open(os.path.join(DATASET_PATH, f"{split}_tag_to_track.json"), mode="w") as io:
        json.dump(tag_to_track, io, indent=4)
    
    with open(os.path.join(DATASET_PATH, f"{split}_tags.json"), mode="w") as io:
        json.dump(tags, io, indent=4)

def main():
    msd_dataset = load_dataset("seungheondoh/LP-MusicCaps-MSD")
    all_samples = []
    for split in ['train', 'valid', 'test']:
        if split != "test":
            build_tag_to_track(msd_dataset, split)
        all_samples += msd_dataset[split]
    with poolcontext(processes=multiprocessing.cpu_count()) as pool:
        pool.map(msd_resampler, all_samples)
    print("finish extract")


if __name__ == '__main__':
    main()