""" Training methods for different models.
"""
import torch
import utils
from torch import nn, optim
from streams.stream_data import WOSStream
from models.wos_classifier import LSTM, LSTMWrapper


def train_wos_batch(epochs=1, lr=0.001, batch_size=utils.BATCH_SIZE, device="cpu"):
    """ Trains a model using batches of data.

    Args:
        epochs (int): number of epochs to go over the dataset
        lr (float): learning rate of the optimizer
        batch_size (int): the batch size
        device (string): the device to run the training on (cpu or gpu)
    """
    stream = WOSStream()
    stream.prepare_for_use()
    model = LSTM(embedding_dim=utils.EMBEDDING_DIM, no_classes=stream.no_classes).to(
        device
    )
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.NLLLoss()

    for epoch in range(epochs):
        # Initialize the loss
        running_loss = 0
        # Start iterating over the dataset
        i = 0
        while stream.has_more_samples():
            # Get the batch from the stream
            if stream.n_remaining_samples() < batch_size:
                x, y, seq_lens = stream.next_sample(stream.n_remaining_samples())
            else:
                x, y, seq_lens = stream.next_sample(batch_size)

            # Move the batch to device
            x.to(device)
            y = torch.from_numpy(y).to(device)
            seq_lens = torch.tensor(seq_lens).to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            predictions, _ = model(x, seq_lens)

            # Loss and backward pass
            loss = criterion(predictions, y)
            loss.backward()
            optimizer.step()

            # Print statistics
            running_loss += loss.item()
            print(i, loss.item())
            if i % 10 == 9:
                # Print every 10 batches
                print("[{}, {}] loss: {}".format(epoch + 1, i + 1, running_loss / 10))
                running_loss = 0
            i += 1

        # Restart the stream
        stream.restart()


def train_wos_stream():
    """ Trains a model using a data stream.

    Returns:

    """
    pass


if __name__ == "__main__":
    train_wos_batch()
