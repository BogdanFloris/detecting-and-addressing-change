"""
Methods that run two streams in parallel in an unsupervised manner.
The trained stream acts as a baseline model from where we get the labels,
and the untrained stream is the one which is compared against the baseline.
"""
import torch
import numpy as np
from utils.metrics import get_metrics


def run_lstm_streams(
    stream_trained,
    stream_untrained,
    model,
    drift_detector,
    batch_size=32,
    print_every=1,
    device="cpu",
):
    """
    Runs the trained stream to collect the labels, and then runs the untrained stream
    to detect changes between the models.

    Args:
        stream_trained (WOSStream): the Web of Science stream on which the model was trained
        stream_untrained (WOSStream): the Web of Science stream to be compared against the trained one
        model (LSTM): the LSTM model to evaluate
        drift_detector: the drift detector used to detect concept drift
        batch_size (int): number of batches
        print_every (int): how often we print
        device (str): cpu or cuda

    Returns:
        a list of accuracies plus, potential warnings or drifts
    """
    i = 0
    # Accuracies list (tuples of accuracy, and drift level)
    trained_accuracies = []
    labels = []
    print("Running trained stream...")
    while stream_trained.has_more_samples():
        # Get the batch from the stream
        if stream_trained.n_remaining_samples() >= batch_size:
            x_, _ = stream_trained.next_sample(batch_size)
        else:
            break

        x, seq_lens = x_
        # Move the batch to device
        x = x.to(device)
        seq_lens = torch.tensor(seq_lens).to(device)

        # Get predictions and add them to labels
        predictions, _ = model((x, seq_lens))
        labels.append(predictions.argmax(dim=1))

        # Print if necessary
        if i % print_every == print_every - 1:
            print("Accuracy: {}".format(1.0))

            # Add to drift detector
            drift_detector.add_element(1 - np.random.uniform(low=0.9, high=1.0))
            if drift_detector.detected_warning_zone():
                trained_accuracies.append((1.0, "W"))
                print("Warning zone")
            elif drift_detector.detected_change():
                trained_accuracies.append((1.0, "D"))
                print("Drift detected")
            else:
                trained_accuracies.append((1.0, "N"))

        i += 1

    i = 0
    running_acc = 0.0
    # Accuracies list (tuples of accuracy, and drift level)
    untrained_accuracies = []
    print("Running untrained stream...")
    while stream_untrained.has_more_samples():
        # Get the batch from the stream
        if stream_untrained.n_remaining_samples() >= batch_size:
            x_, _ = stream_untrained.next_sample(batch_size)
            y = labels[i]
        else:
            break

        x, seq_lens = x_
        # Move the batch to device
        x = x.to(device)
        seq_lens = torch.tensor(seq_lens).to(device)

        # Get predictions and accuracy
        predictions, _ = model((x, seq_lens))
        metrics = get_metrics(
            labels=y.detach().numpy(),
            predictions=predictions,
            no_labels=stream_untrained.n_classes,
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
                untrained_accuracies.append((accuracy, "W"))
                print("Warning zone")
            elif drift_detector.detected_change():
                untrained_accuracies.append((accuracy, "D"))
                print("Drift detected")
            else:
                untrained_accuracies.append((accuracy, "N"))

        i += 1

    return trained_accuracies, untrained_accuracies


def run_nb_streams(
    stream_trained,
    stream_untrained,
    model,
    drift_detector,
    batch_size=32,
    print_every=1,
    device="cpu",
):
    """
    Runs the trained stream to collect the labels, and then runs the untrained stream
    to detect changes between the models.

    Args:
        stream_trained (WOSStream): the Web of Science stream on which the model was trained
        stream_untrained (WOSStream): the Web of Science stream to be compared against the trained one
        model (NaiveBayes): the Naive Bayes model to evaluate
        drift_detector: the drift detector used to detect concept drift
        batch_size (int): number of batches
        print_every (int): how often we print
        device (str): cpu or cuda

    Returns:
        a list of accuracies plus, potential warnings or drifts

    """
    i = 0
    # Accuracies list (tuples of accuracy, and drift level)
    trained_accuracies = []
    labels = []
    print("Running trained stream...")
    while stream_trained.has_more_samples():
        # Get the batch from the stream
        if stream_trained.n_remaining_samples() >= batch_size:
            x_, _ = stream_trained.next_sample(batch_size)
        else:
            break

        # Unpack x_ (we do not need the sequence lengths for NB)
        x = x_[0].numpy()
        # Take the maximum over the axis 1
        x = np.amax(x, axis=1)

        # Get the predictions and metrics
        y_pred = model.predict(x)
        labels.append(y_pred)

        # Print if necessary
        if i % print_every == print_every - 1:
            print("Accuracy: {}".format(1.0))

            # Add to drift detector
            drift_detector.add_element(1 - np.random.uniform(low=0.9, high=1.0))
            if drift_detector.detected_warning_zone():
                trained_accuracies.append((1.0, "W"))
                print("Warning zone")
            elif drift_detector.detected_change():
                trained_accuracies.append((1.0, "D"))
                print("Drift detected")
            else:
                trained_accuracies.append((1.0, "N"))

        i += 1

    i = 0
    running_acc = 0.0
    # Accuracies list (tuples of accuracy, and drift level)
    untrained_accuracies = []
    print("Running untrained stream...")
    while stream_untrained.has_more_samples():
        # Get the batch from the stream
        if stream_untrained.n_remaining_samples() >= batch_size:
            x_, _ = stream_untrained.next_sample(batch_size)
            y = labels[i]
        else:
            break

        # Unpack x_ (we do not need the sequence lengths for NB)
        x = x_[0].numpy()
        # Take the maximum over the axis 1
        x = np.amax(x, axis=1)

        # Get the predictions and metrics
        y_pred = model.predict(x)
        metrics = get_metrics(
            labels=y, predictions=y_pred, no_labels=stream_untrained.n_classes
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
                untrained_accuracies.append((accuracy, "W"))
                print("Warning zone")
            elif drift_detector.detected_change():
                untrained_accuracies.append((accuracy, "D"))
                print("Drift detected")
            else:
                untrained_accuracies.append((accuracy, "N"))

        i += 1

    return trained_accuracies, untrained_accuracies
