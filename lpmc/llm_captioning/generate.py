import os
import openai
import argparse
import json
import argparse
import random
from dotenv import load_dotenv
from tqdm import tqdm
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from time import sleep

def api_helper(instance):
    text = instance['text']
    split = instance['split']
    inputs = instance['inputs']
    prompt = instance['prompt']
    dataset_type = instance['dataset_type']
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
                {"role": "user", "content": inputs}
            ]
    )
    results = completion['choices'][0]['message']['content']

    print("query: ")
    print(inputs)
    print("-"*10)
    print("results: ")
    print(results)
    print("="*15)
    os.makedirs(f"./samples/{dataset_type}/{prompt}/{split}", exist_ok=True)
    with open(f"./samples/{dataset_type}/{prompt}/{split}/{instance['_id']}.txt", 'w') as file:
        file.write(results)

class OpenAIGpt:
    def __init__(self, split, prompt, dataset_type, n_iter=True):
        load_dotenv()    
        self.split = split
        self.prompt = prompt
        self.dataset_type = dataset_type
        if self.dataset_type == "msd":
            self.annotation= json.load(open("./dataset/ecals_annotation/annotation.json", 'r'))
            self.track_split= json.load(open("./dataset/ecals_annotation/ecals_track_split.json", 'r'))
        elif self.dataset_type == "mtat":
            self.annotation = json.load(open("./dataset/mtat/codified_annotation.json", 'r'))
            self.track_split = json.load(open("./dataset/mtat/codified_track_split.json", 'r'))
        elif self.dataset_type == "musiccaps":
            self.annotation = json.load(open("./dataset/musiccaps/annotation.json", 'r'))
            self.track_split = json.load(open("./dataset/musiccaps/track_split.json", 'r'))
        self.prompt_dict = {
            "writing": {
                "singular":"write a song description sentence including the following single attribute.",
                "plural":"write a song description sentence including the following attributes.",
                },
            "summary": {
                "singular":"write a single sentence that summarize a song with the following single attribute. Don't write artist name or album name.",
                "plural":"write a single sentence that summarize a song with the following attributes. Don't write artist name or album name.",
                },
            "paraphrase": {
                "singular":"write a song description sentence including the following single attribute. creative paraphrasing is acceptable.",
                "plural":"write a song description sentence including the following attributes. creative paraphrasing is acceptable.",
                },
            "attribute_prediction": {
                "singular":"write the answer as a python dictionary with new_attribute and description as keys. for new_attribute, write new attributes with high co-occurrence with the following single attribute. for description, write a song description sentence including the single attribute and new attribute.",
                "plural":"write the answer as a python dictionary with new_attribute and description as keys. for new_attribute, write new attributes with high co-occurrence with the following attributes. for description, write a song description sentence including the following attributes and new attributes.",
                }
            }
        if split == "TRAIN":
            if self.dataset_type == "msd":
                train_track = self.track_split['train_track'] + self.track_split['extra_track']
            else:
                target_track = self.track_split['train_track']
        elif split == "VALID":
            target_track = self.track_split['valid_track']
        else:
            target_track = self.track_split['test_track']

        if n_iter:
            self.get_already_download()
            target_track = list(set(target_track).difference(self.already_download))
        self.fl_dict = {i : self.annotation[i] for i in target_track}
        
    def get_already_download(self):
        save_path = f"./samples/results/{self.dataset_type}/{self.prompt}/{self.split}"
        self.already_download = set([i.replace(".txt", "")for i in os.listdir(save_path)])
        print("already_download: ", len(self.already_download))

    def run(self):
        openai.api_key = os.getenv("OPENAI_API_KEY")
        inputs = []
        
        if len(self.fl_dict) > 0:
            for _id, instance in self.fl_dict.items():
                instance['_id'] = _id
                instance['split'] = self.split
                instance['prompt'] = self.prompt
                instance['dataset_type'] = self.dataset_type
                if self.dataset_type == "msd":
                    tags = instance["tag"]
                elif self.dataset_type == "mtat":
                    tags = instance['extra_tag']
                elif self.dataset_type == "musiccaps":
                    tags = instance['aspect_list']
                text = ", ".join(tags)
                instance["text"] = text
                if len(tags) > 1:
                    instruction = self.prompt_dict[self.prompt]["plural"]
                elif len(tags) == 0:
                    continue
                else:
                    instruction = self.prompt_dict[self.prompt]["singular"]
                instance["inputs"] = f'{instruction} \n {text}'
                inputs.append(instance)
            with ThreadPoolExecutor() as pool:
                tqdm(pool.map(api_helper, inputs))
            print("finish")
        else:
            print("already finished")
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_type", default="musiccaps", type=str)
    parser.add_argument("--split", default="TRAIN", type=str)
    parser.add_argument("--prompt", default="writing", type=str)
    parser.add_argument("--n_iter", default=False, type=bool)
    args = parser.parse_args()

    openai_gpt = OpenAIGpt(
        split = args.split, 
        prompt = args.prompt, 
        dataset_type = args.dataset_type,
        n_iter = args.n_iter
        )
    openai_gpt.run()