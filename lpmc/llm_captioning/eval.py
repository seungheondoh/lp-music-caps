import os
import random
import argparse
import json
import torch
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from lpmc.utils.metrics import bleu, meteor, rouge, bertscore, vocab_novelty, caption_novelty

def _apply_template(tag_concat):
    return f"the music is characterized by {tag_concat}"

def baseline_generation(dataset, prediction_col):
    predictions = []
    for item in tqdm(dataset):
        tag_list = item['aspect_list']
        random.shuffle(tag_list)
        tag_concat = ", ".join(tag_list)
        if prediction_col == "baseline_tagconcat":
            predictions.append(tag_concat)
        elif prediction_col == "baseline_template":
            predictions.append(_apply_template(tag_concat))
    return predictions

def inference_parsing(dataset, prediction_col):
    ground_truths = [i['caption_ground_truth'] for i in dataset]
    if "baseline" in prediction_col:
        predictions = baseline_generation(dataset, prediction_col)
    else:
        predictions = [i[prediction_col] for i in dataset]
    return predictions, ground_truths

def main(args):
    dataset = load_dataset("seungheondoh/LP-MusicCaps-MC")
    train_data = [i for i in dataset['train'] if i['is_crawled']]
    test_data = [i for i in dataset['test'] if i['is_crawled']]
    _, tr_ground_truths = inference_parsing(train_data, args.prediction_col)
    predictions, ground_truths = inference_parsing(test_data, args.prediction_col)
    inference = [{"prediction":pre,"ground_truth":gt} for pre,gt in zip(predictions, ground_truths)]
    length_avg = np.mean([len(cap.split()) for cap in predictions])
    length_std = np.std([len(cap.split()) for cap in predictions])
    
    vocab_size, vocab_novel_score = vocab_novelty(predictions, tr_ground_truths)
    cap_novel_score = caption_novelty(predictions, tr_ground_truths)
    results = {
        "bleu1": bleu(predictions, ground_truths, order=1),
        "bleu2": bleu(predictions, ground_truths, order=2),
        "bleu3": bleu(predictions, ground_truths, order=3),
        "bleu4": bleu(predictions, ground_truths, order=4),
        "meteor_1.0": meteor(predictions, ground_truths), # https://github.com/huggingface/evaluate/issues/115
        "rougeL": rouge(predictions, ground_truths),
        "bertscore": bertscore(predictions, ground_truths),
        "vocab_size": vocab_size, 
        "vocab_novelty": vocab_novel_score, 
        "caption_novelty": cap_novel_score,
        "length_avg": length_avg,
        "length_std": length_std
    }
    os.makedirs(os.path.join(args.save_dir, args.prediction_col), exist_ok=True)
    with open(os.path.join(args.save_dir, args.prediction_col, f"results.json"), mode="w") as io:
        json.dump(results, io, indent=4)
    with open(os.path.join(args.save_dir, args.prediction_col, f"inference.json"), mode="w") as io:
        json.dump(inference, io, indent=4)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", default="./exp", type=str)
    parser.add_argument("--prediction_col", 
            default="caption_writing", 
            # choices=['caption_writing', 'caption_summary', 'caption_paraphrase', 'caption_attribute_prediction'], 
            type=str
    )
    args = parser.parse_args()
    main(args=args)