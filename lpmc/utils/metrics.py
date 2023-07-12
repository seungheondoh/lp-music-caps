"""Placeholder for metrics."""
from functools import partial
import evaluate
import numpy as np
import torch
import torchmetrics.retrieval as retrieval_metrics
# CAPTIONING METRICS
def bleu(predictions, ground_truths, order):
    bleu_eval = evaluate.load("bleu")
    return bleu_eval.compute(
        predictions=predictions, references=ground_truths, max_order=order
    )["bleu"]

def meteor(predictions, ground_truths):
    # https://github.com/huggingface/evaluate/issues/115
    meteor_eval = evaluate.load("meteor")
    return meteor_eval.compute(predictions=predictions, references=ground_truths)[
        "meteor"
    ]


def rouge(predictions, ground_truths):
    rouge_eval = evaluate.load("rouge")
    return rouge_eval.compute(predictions=predictions, references=ground_truths)[
        "rougeL"
    ]


def bertscore(predictions, ground_truths):
    bertscore_eval = evaluate.load("bertscore")
    score = bertscore_eval.compute(
        predictions=predictions, references=ground_truths, lang="en"
    )["f1"]
    return np.mean(score)


def vocab_diversity(predictions, references):
    train_caps_tokenized = [
        train_cap.translate(str.maketrans("", "", string.punctuation)).lower().split()
        for train_cap in references
    ]
    gen_caps_tokenized = [
        gen_cap.translate(str.maketrans("", "", string.punctuation)).lower().split()
        for gen_cap in predictions
    ]
    training_vocab = Vocabulary(train_caps_tokenized, min_count=2).idx2word
    generated_vocab = Vocabulary(gen_caps_tokenized, min_count=1).idx2word

    return len(generated_vocab) / len(training_vocab)


def vocab_novelty(predictions, tr_ground_truths):
    predictions_token, tr_ground_truths_token = [], []
    for gen, ref in zip(predictions, tr_ground_truths):
        predictions_token.extend(gen.lower().replace(",","").replace(".","").split())
        tr_ground_truths_token.extend(ref.lower().replace(",","").replace(".","").split())

    predictions_vocab = set(predictions_token)
    new_vocab = predictions_vocab.difference(set(tr_ground_truths_token))
    
    vocab_size = len(predictions_vocab)
    novel_v = len(new_vocab) / vocab_size
    return vocab_size, novel_v

def caption_novelty(predictions, tr_ground_truths):
    unique_pred_captions = set(predictions)
    unique_train_captions = set(tr_ground_truths)

    new_caption = unique_pred_captions.difference(unique_train_captions)
    novel_c = len(new_caption) / len(unique_pred_captions)
    return novel_c

def metric_1(predictions, ground_truths) -> float:
    """Computes metric_1 score.
    Args:
        predictions: A list of predictions.
        ground_truths: A list of ground truths.
    Returns:
        metric_1: A float number, the metric_1 score.
    """
    return 0.0


# RETRIEVAL METRICS
def _prepare_torchmetrics_input(scores, query2target_idx):
    target = [
        [i in target_idxs for i in range(len(scores[0]))]
        for query_idx, target_idxs in query2target_idx.items()
    ]
    indexes = torch.arange(len(scores)).unsqueeze(1).repeat((1, len(target[0])))
    return torch.as_tensor(scores), torch.as_tensor(target), indexes


def _call_torchmetrics(
    metric: retrieval_metrics.RetrievalMetric, scores, query2target_idx, **kwargs
):
    preds, target, indexes = _prepare_torchmetrics_input(scores, query2target_idx)
    return metric(preds, target, indexes=indexes, **kwargs).item()


def recall(predicted_scores, query2target_idx, k: int) -> float:
    """Compute retrieval recall score at cutoff k.

    Args:
        predicted_scores: N x M similarity matrix
        query2target_idx: a dictionary with
            key: unique query idx
            values: list of target idx
        k: number of top-k results considered
    Returns:
        average score of recall@k
    """
    recall_metric = retrieval_metrics.RetrievalRecall(k=k)
    return _call_torchmetrics(recall_metric, predicted_scores, query2target_idx)


def mean_average_precision(predicted_scores, query2target_idx) -> float:
    """Compute retrieval mean average precision (MAP) score at cutoff k.

    Args:
        predicted_scores: N x M similarity matrix
        query2target_idx: a dictionary with
            key: unique query idx
            values: list of target idx
    Returns:
        MAP@k score
    """
    map_metric = retrieval_metrics.RetrievalMAP()
    return _call_torchmetrics(map_metric, predicted_scores, query2target_idx)


def mean_reciprocal_rank(predicted_scores, query2target_idx) -> float:
    """Compute retrieval mean reciprocal rank (MRR) score.

    Args:
        predicted_scores: N x M similarity matrix
        query2target_idx: a dictionary with
            key: unique query idx
            values: list of target idx
    Returns:
        MRR score
    """
    mrr_metric = retrieval_metrics.RetrievalMRR()
    return _call_torchmetrics(mrr_metric, predicted_scores, query2target_idx)