import torch


def accuracy(labels, predictions):
    """ Computes the accuracy given the labels and predictions.

    Args:
        labels (tensor): the correct labels
        predictions (tensor): the predictions of the model

    Returns:
        the accuracy
    """
    return (labels == predictions.argmax(dim=1)).type(torch.FloatTensor).mean()
