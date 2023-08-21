import os
import json
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from datasets import load_dataset

class MSD_Balanced_Dataset(Dataset):
    def __init__(self, data_path, split, caption_type, sr=16000, duration=10, audio_enc="wav"):
        self.data_path = data_path
        self.split = split
        self.audio_enc = audio_enc
        self.n_samples = int(sr * duration)
        self.caption_type = caption_type
        self.dataset = load_dataset("seungheondoh/LP-MusicCaps-MSD")
        self.get_split()

    def get_split(self):
        self.tags = json.load(open(os.path.join(self.data_path, "msd", f"{self.split}_tags.json"), 'r'))
        self.tag_to_track = json.load(open(os.path.join(self.data_path, "msd", f"{self.split}_tag_to_track.json"), 'r'))
        self.annotation = {instance['track_id'] : instance for instance in self.dataset[self.split]}

    def load_caption(self, item):
        caption_pool = []
        if (self.caption_type in "write") or (self.caption_type == "lp_music_caps"):
            caption_pool.append(item['caption_writing'])
        if (self.caption_type in "summary") or (self.caption_type == "lp_music_caps"):
            caption_pool.append(item['caption_summary'])
        if (self.caption_type in "creative") or (self.caption_type == "lp_music_caps"):
            caption_pool.append(item['caption_paraphrase'])
        if (self.caption_type in "predict") or (self.caption_type == "lp_music_caps"):
            caption_pool.append(item['caption_attribute_prediction'])
        # randomly select one caption from multiple captions
        sampled_caption = random.choice(caption_pool)
        return sampled_caption
    
    def load_audio(self, audio_path, file_type):
        audio = np.load(audio_path, mmap_mode='r')
        if len(audio.shape) == 2:
            audio = audio.squeeze(0)
        input_size = int(self.n_samples)
        if audio.shape[-1] < input_size:
            pad = np.zeros(input_size)
            pad[:audio.shape[-1]] = audio
            audio = pad
        random_idx = random.randint(0, audio.shape[-1]-self.n_samples)
        audio_tensor = torch.from_numpy(np.array(audio[random_idx:random_idx+self.n_samples]).astype('float32'))
        return audio_tensor

    def __getitem__(self, index):
        tag = random.choice(self.tags) # uniform random sample tag
        fname = random.choice(self.tag_to_track[tag])  # uniform random sample track
        item = self.annotation[fname]
        track_id = item['track_id']
        gt_caption = "" # no ground turhth
        text = self.load_caption(item)
        audio_path = os.path.join(self.data_path,'msd','npy', item['path'].replace(".mp3", ".npy"))
        audio_tensor = self.load_audio(audio_path, file_type=".npy")
        return fname, gt_caption, text, audio_tensor

    def __len__(self):
        return 2**13