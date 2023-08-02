import os
import argparse
import json
import numpy as np
from datasets import load_dataset
from lpmc.utils.metrics import bleu, meteor, rouge, bertscore, vocab_novelty, caption_novelty

def inference_parsing(dataset, args):
    ground_truths = [i['caption_ground_truth'] for i in dataset]
    inference = json.load(open(os.path.join(args.save_dir, args.framework, args.caption_type, 'inference_temp.json'), 'r'))
    id2pred = {item['audio_id']:item['predictions'] for item in inference.values()}
    predictions = [id2pred[i['fname']] for i in dataset]
    return predictions, ground_truths

def main(args):
    dataset = load_dataset("seungheondoh/LP-MusicCaps-MC")
    train_data = [i for i in dataset['train'] if i['is_crawled']]
    test_data = [i for i in dataset['test'] if i['is_crawled']]
    tr_ground_truths = [i['caption_ground_truth'] for i in train_data]
    predictions, ground_truths = inference_parsing(test_data, args)
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
    os.makedirs(os.path.join(args.save_dir, args.framework, args.caption_type), exist_ok=True)
    with open(os.path.join(args.save_dir, args.framework, args.caption_type, f"results.json"), mode="w") as io:
        json.dump(results, io, indent=4)
    print(results)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", default="./exp", type=str)
    parser.add_argument("--framework", default="supervised", type=str)
    parser.add_argument("--caption_type", default="gt", type=str)
    args = parser.parse_args()
    main(args=args)