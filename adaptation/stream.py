""" Method to run a stream with a mapping.
"""
import torch
from streams.stream_data import WOSStream
from models.wos_classifier import LSTM
from adaptation.procrustes import Procrustes
from utils.metrics import get_metrics


def run_stream_with_mapping(
    stream, model, mapping, batch_size=1, print_every=1, device="cpu",
):
    """ Runs a stream with a mapping to convert from the stream's
    inputs embeddings space to the embedding space outputted by
    the mapping.

    Args:
        stream (WOSStream): the Web of Science stream to be run
        model (LSTM): the LSTM model to evaluate
        mapping (Procrustes): the mapping used to change embedding spaces
        batch_size (int): number of batches
        print_every (int): how often we print
        device (str): cpu or cuda

    Returns:
        a list of accuracies

    """
    # Initialize variables for tracking
    i = 0
    running_acc = 0.0
    accuracies = []
    mapping = torch.tensor(mapping.mapping)

    # Run stream
    while stream.has_more_samples():
        # Get the batch from the stream
        if stream.n_remaining_samples() >= batch_size:
            x_, y = stream.next_sample(batch_size)
        else:
            break

        x, seq_lens = x_
        # Put in the mapping to transform to the other embedding space
        x = x.matmul(mapping.type(torch.FloatTensor)).to(device)
        y = torch.from_numpy(y).to(device)
        seq_lens = torch.tensor(seq_lens).to(device)

        # Get predictions and accuracy
        predictions, _ = model((x, seq_lens))
        metrics = get_metrics(
            labels=y, predictions=predictions, no_labels=stream.n_classes
        )
        accuracy = metrics["accuracy"]

        # Print if necessary
        running_acc += accuracy
        if i % print_every == print_every - 1:
            print("Accuracy: {}".format(running_acc / print_every))
            running_acc = 0.0

        i += 1

    return accuracies
