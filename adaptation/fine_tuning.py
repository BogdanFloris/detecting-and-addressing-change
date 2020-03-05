""" Methods to fine tune a model to recover
from small abrupt concept drift or gradual concept drift.
"""
import torch
from torch import nn, optim
from sklearn.metrics import accuracy_score
import utils


def fine_tune(
    model,
    stream,
    no_batches,
    lr=0.001,
    batch_size=utils.BATCH_SIZE,
    print_every=1,
    device="cpu",
):
    """ Fine tunes an LSTM model with a few samples.

    Args:
        model (LSTM): the LSTM model to be fine tuned
        stream (WOSStream): the stream from where we get samples
        no_batches (int): the number of batches fed to the model
        lr (float): the learning rate
        batch_size (int): the batch size
        print_every (int): how often we print
        device (str): cpu or cuda

    Returns:
        the accuracies of the trained model

    """
    # Switch the model to train mode
    model.train()

    # Initialize the optimizer and the criterion
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.NLLLoss()

    # Initialize running loss and accuracy
    running_loss = 0.0
    running_accuracy = 0.0
    # Accuracies
    accuracies = []
    for i in range(no_batches):
        # Get the batch from the stream
        if stream.n_remaining_samples() >= batch_size:
            x_, y = stream.next_sample(batch_size)
        else:
            stream.restart()
            continue

        # Move the batch to device
        x, seq_lens = x_
        x = x.to(device)
        y = torch.from_numpy(y).to(device)
        seq_lens = torch.tensor(seq_lens).to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        predictions, _ = model((x, seq_lens))

        # Loss and backward pass
        loss = criterion(predictions, y)
        loss.backward()
        optimizer.step()

        # Print statistics
        running_loss += loss.item()
        running_accuracy += accuracy_score(y.cpu(), predictions.argmax(dim=1).cpu())

        if i % print_every == print_every - 1 or not stream.has_more_samples():
            denominator = (
                print_every
                if stream.has_more_samples()
                else (stream.n_samples // batch_size + 1) % print_every
            )
            train_loss = running_loss / denominator
            train_acc = running_accuracy / denominator

            # Print every 10 batches
            print(
                "[{}/{} batches] train loss: {:.4f}, "
                "train accuracy: {:.4f}".format(
                    i + 1, no_batches, train_loss, train_acc,
                )
            )

            accuracies.append(train_acc)
            running_loss = 0.0
            running_accuracy = 0.0

    model.eval()
    return accuracies
