import os
import json
import random
import pickle
import numpy as np
import pandas as pd
import torch
from re import sub
from typing import Callable, List, Dict, Any
from torch.utils.data import Dataset, DataLoader
from mcb.preprocessing.audio_utils import load_audio, STR_CH_FIRST

class MSD_Balanced_Dataset(Dataset):
    def __init__(self, data_path, split, text_type, sr=16000, duration=10, audio_enc="wav"):
        self.data_path = data_path
        self.split = split
        self.audio_enc = audio_enc
        self.n_samples = int(sr * duration)
        self.text_type = text_type
        self.msd_to_id = pickle.load(open(os.path.join(self.data_path, "msd", "MSD_id_to_7D_id.pkl"), 'rb'))
        self.id_to_path = pickle.load(open(os.path.join(self.data_path, "msd", "7D_id_to_path.pkl"), 'rb'))
        self.annotation = json.load(open(os.path.join(self.data_path, "msd", "annotation.json"), 'r'))
        self.tags = json.load(open(os.path.join(self.data_path, "msd", "ecals_tags.json"), 'r'))
        self.get_split()
        self.get_caption_pool()

    def get_split(self):
        track_split = json.load(open(os.path.join(self.data_path, "msd", "ecals_track_split.json"), "r"))
        multiquery_samples = json.load(open(os.path.join(self.data_path, "msd", "multiquery_samples.json"), "r"))
        if self.split == "TRAIN":
            self.fl = track_split['train_track'] + track_split['valid_track'] + track_split['extra_track']
            self.tag_to_track = json.load(open(os.path.join(self.data_path, "msd", "TRAIN_tag_to_track.json"), 'r'))
            self.valid_ttt = json.load(open(os.path.join(self.data_path, "msd", "VALID_tag_to_track.json"), 'r'))
            for _id, track_list in self.tag_to_track.items():
                self.tag_to_track[_id] =  track_list + self.valid_ttt[_id]
        # elif self.split == "VALID":
        #     self.fl = track_split['valid_track']
        elif self.split == "TEST":
            self.fl = list(multiquery_samples.keys())
            self.tag_to_track = json.load(open(os.path.join(self.data_path, "msd", "TEST_tag_to_track.json"), 'r'))
        else:
            raise ValueError(f"Unexpected split name: {self.split}")

    def _get_id2cap(self, text_type):
        id2cap = {}
        train_case = json.load(open(os.path.join(self.data_path, "msd", 'lpmc', text_type, f"TRAIN_dict.json"), "r"))
        valid_case = json.load(open(os.path.join(self.data_path, "msd", 'lpmc', text_type, f"VALID_dict.json"), "r"))
        id2cap.update(train_case)
        id2cap.update(valid_case)
        return id2cap
        
    def get_caption_pool(self):
        # baseline
        if self.text_type in "tag_concat":
            self.tag_concat = self._get_id2cap(self.text_type)
        if self.text_type in "template":
            self.template = self._get_id2cap(self.text_type)
        if self.text_type in "k2c":
            self.k2c = self._get_id2cap(self.text_type)
        # proposed caption
        if (self.text_type in "write") or (self.text_type == "all"):
            self.llm_write = self._get_id2cap("write")
        if (self.text_type in "summary") or (self.text_type == "all"):
            self.llm_summary = self._get_id2cap("summary")
        if (self.text_type in "creative") or (self.text_type == "all"):
            self.llm_creative = self._get_id2cap("creative")
        if (self.text_type in "predict") or (self.text_type == "all"):
            self.llm_predict = self._get_id2cap("predict")

    def load_caption(self, fname):
        caption_pool = []
        if self.text_type in "tag_concat":
            pseudo_caption = self.tag_concat[fname]
            caption_pool.append(pseudo_caption)
        if self.text_type in "template":
            pseudo_caption = self.template[fname]
            caption_pool.append(pseudo_caption)
        if self.text_type in "k2c":
            pseudo_caption = self.k2c[fname]
            caption_pool.append(pseudo_caption)
        if (self.text_type in "write") or (self.text_type == "all"):
            pseudo_caption = self.llm_write[fname]
            caption_pool.append(pseudo_caption)
        if (self.text_type in "summary") or (self.text_type == "all"):
            pseudo_caption = self.llm_summary[fname]
            caption_pool.append(pseudo_caption)
        if (self.text_type in "creative") or (self.text_type == "all"):
            pseudo_caption = self.llm_creative[fname]
            caption_pool.append(pseudo_caption)
        if (self.text_type in "predict") or (self.text_type == "all"):
            pseudo_caption = self.llm_predict[fname]
            caption_pool.append(pseudo_caption)
        # randomly select one caption from multiple captions
        sampled_caption = random.choice(caption_pool)
        return sampled_caption
    
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
        tag = random.choice(self.tags)
        fname = random.choice(self.tag_to_track[tag])
        item = self.annotation[fname]
        track_id = item['track_id']
        gt_caption = "" # no ground turhth
        text = self.load_caption(fname)
        audio_path = os.path.join(self.data_path,'msd','npy', self.id_to_path[self.msd_to_id[track_id]].replace(".mp3", ".npy"))
        audio_tensor = self.load_audio(audio_path, file_type=audio_path[-4:])
        return fname, gt_caption, text, audio_tensor

    def __len__(self):
        return 2**13