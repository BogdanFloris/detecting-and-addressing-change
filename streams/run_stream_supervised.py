"""
Methods that run a stream against a model and drift detector.
It gets the labels from the stream (supervised).
"""
import torch
import numpy as np
from utils.metrics import get_metrics


def run_stream_lstm(
    stream,
    model,
    drift_detector,
    batch_size=1,
    print_every=1,
    noise_stds=None,
    warm_start=None,
    device="cpu",
):
    """
    Runs a stream on the LSTM model using the given drift detector.

    Args:
        stream (WOSStream): the Web of Science stream to be run
        model (LSTM): the LSTM model to evaluate
        drift_detector: the drift detector used to detect concept drift
        batch_size (int): number of batches
        print_every (int): how often we print
        noise_stds (list): a list of standard deviations for the gradual noise.
            If none, no noise is added
        warm_start (int): after which batch we start adding noise.
        device (str): cpu or cuda

    Returns:
        a list of accuracies plus, potential warnings or drifts
    """
    i = 0
    running_acc = 0.0
    # Accuracies list (tuples of accuracy, and drift level)
    accuracies = []
    while stream.has_more_samples():
        # Get the batch from the stream
        if stream.n_remaining_samples() >= batch_size:
            x_, y = stream.next_sample(batch_size)
        else:
            break

        x, seq_lens = x_
        # Add noise if we have standard deviations
        if i >= warm_start and noise_stds is not None:
            print("Adding noise")
            std = torch.zeros_like(x) + noise_stds[i - warm_start]
            noise = torch.normal(0, std)
            x = x + noise

        # Move the batch to device
        x = x.to(device)
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

            # Add to drift detector
            drift_detector.add_element(1 - accuracy)
            if drift_detector.detected_warning_zone():
                accuracies.append((accuracy, "W"))
                print("Warning zone")
            elif drift_detector.detected_change():
                accuracies.append((accuracy, "D"))
                print("Drift detected")
            else:
                accuracies.append((accuracy, "N"))

        i += 1

    return accuracies


def run_stream_nb(
    stream,
    model,
    drift_detector,
    batch_size=1,
    print_every=1,
    noise_stds=None,
    warm_start=None,
    device="cpu",
):
    """
    Runs a stream on the LSTM model using the given drift detector.

    Args:
        stream (WOSStream): the Web of Science stream to be run
        model (NaiveBayes): the Naive Bayes model to evaluate
        drift_detector: the drift detector used to detect concept drift
        batch_size (int): number of batches
        print_every (int): how often we print
        noise_stds (list): a list of standard deviations for the gradual noise.
            If none, no noise is added
        warm_start (int): after which batch we start adding noise.
        device (str): cpu or cuda

    Returns:
        a list of accuracies plus, potential warnings or drifts
    """
    i = 0
    running_acc = 0.0
    # Accuracies list (tuples of accuracy, and drift level)
    accuracies = []
    while stream.has_more_samples():
        # Get the batch from the stream
        if stream.n_remaining_samples() >= batch_size:
            x_, y = stream.next_sample(batch_size)
        else:
            break

        # Unpack x_ (we do not need the sequence lengths for NB)
        x = x_[0].numpy()
        # Take the maximum over the axis 1
        x = np.amax(x, axis=1)
        # Add noise if we have standard deviations
        if i >= warm_start and noise_stds is not None:
            print("Adding noise")
            noise = np.random.normal(0, noise_stds[i - warm_start], x.shape)
            x = x + noise

        # Get the predictions and metrics
        y_pred = model.predict(x)
        metrics = get_metrics(labels=y, predictions=y_pred, no_labels=stream.n_classes)
        accuracy = metrics["accuracy"]

        # Print if necessary
        running_acc += accuracy
        if i % print_every == print_every - 1:
            print("Accuracy: {}".format(running_acc / print_every))
            running_acc = 0.0

            # Add to drift detector
            drift_detector.add_element(1 - accuracy)
            if drift_detector.detected_warning_zone():
                accuracies.append((accuracy, "W"))
                print("Warning zone")
            elif drift_detector.detected_change():
                accuracies.append((accuracy, "D"))
                print("Drift detected")
            else:
                accuracies.append((accuracy, "N"))

        i += 1

    return accuracies
