import os
import random
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from torch.utils.data import Dataset
from lpmc.utils.audio_utils import load_audio, STR_CH_FIRST

class MC_Dataset(Dataset):
    def __init__(self, data_path, split, caption_type, sr=16000, duration=10, audio_enc="wav"):
        self.data_path = data_path
        self.split = split
        self.caption_type = caption_type
        self.audio_enc = audio_enc
        self.n_samples = int(sr * duration)
        self.annotation = load_dataset("seungheondoh/LP-MusicCaps-MC")
        self.get_split()

    def get_split(self):
        if self.split == "train":
            self.fl = [i for i in self.annotation[self.split] if i['is_crawled']]
        elif self.split == "test":
            self.fl = [i for i in self.annotation[self.split] if i['is_crawled']]
        else:
            raise ValueError(f"Unexpected split name: {self.split}")

    def load_audio(self, audio_path, file_type):
        if file_type == ".npy":
            audio = np.load(audio_path, mmap_mode='r')
        else:
            audio, _ = load_audio(
                path=audio_path,
                ch_format= STR_CH_FIRST,
                sample_rate= self.sr,
                downmix_to_mono= True
            )
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
        item = self.fl[index]
        fname = item['fname']
        text = item['caption_ground_truth']
        audio_path = os.path.join(self.data_path, "music_caps", 'npy', fname + ".npy")
        audio_tensor = self.load_audio(audio_path, file_type=audio_path[-4:])
        return fname, text, audio_tensor

    def __len__(self):
        return len(self.fl)