from typing import Dict
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def get_metrics(predictions, labels, no_labels=11) -> Dict[str, float]:
    """ Computes metrics for the given predictions and labels.

    Args:
        predictions (tensor): the predictions of the model
        labels (tensor): the correct labels
        no_labels (int): number of true labels

    Returns:
        dictionary from the name of the metrics to the value of the metric
    """
    predictions = predictions.argmax(dim=1)
    all_labels = np.arange(no_labels)
    metrics = {
        "accuracy": accuracy_score(labels, predictions),
        "precision": precision_score(
            labels, predictions, labels=all_labels, average="macro", zero_division=0
        ),
        "recall": recall_score(
            labels, predictions, labels=all_labels, average="macro", zero_division=0
        ),
        "macro_f1": f1_score(
            labels, predictions, labels=all_labels, average="macro", zero_division=0
        ),
    }
    return metrics
