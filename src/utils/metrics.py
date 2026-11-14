import numpy as np
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

def compute_metrics_single(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=1)

    return {
        "accuracy": accuracy_score(labels, preds),
        "precision_macro": precision_score(labels, preds, average="macro", zero_division=0),
        "recall_macro": recall_score(labels, preds, average="macro", zero_division=0),
        "f1_macro": f1_score(labels, preds, average="macro", zero_division=0),
    }

def compute_metrics_multi(eval_pred):
    logits, labels = eval_pred
    logits = np.asarray(logits, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int32)

    probs = 1.0 / (1.0 + np.exp(-logits))
    preds = (probs > 0.5).astype(np.int32)

    return {
        "accuracy": accuracy_score(labels, preds),
        "precision_macro": precision_score(labels, preds, average="macro", zero_division=0),
        "recall_macro": recall_score(labels, preds, average="macro", zero_division=0),
        "f1_macro": f1_score(labels, preds, average="macro", zero_division=0),
    }