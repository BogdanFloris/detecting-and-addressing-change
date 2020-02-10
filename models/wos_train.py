""" Training methods for different models.
"""
import os
import torch
import utils
from pathlib import Path
from torch import nn, optim
from skmultiflow.evaluation import EvaluateHoldout
from streams.stream_data import WOSStream
from models.wos_classifier import LSTM, LSTMStream
from constants.transformers import TransformerModel
from utils.metrics import accuracy


PATH = os.path.join(Path(__file__).parents[1], "assets/models")
if not os.path.isdir(PATH):
    os.makedirs(PATH)


def train_wos_batch(
    epochs=1,
    lr=0.001,
    batch_size=utils.BATCH_SIZE,
    transform=True,
    transformer_model=TransformerModel.BERT,
    print_every=10,
    load_checkpoint=False,
    device="cpu",
):
    """ Trains a model using batches of data.

    Args:
        epochs (int): number of epochs to go over the dataset
        lr (float): learning rate of the optimizer
        batch_size (int): the batch size
        transform (bool): transform the dataset or not
        transformer_model (TransformerModel): the transformer model to use
        print_every (int): print stats parameter
        load_checkpoint (bool): load a checkpoint or not
        device (string): the device to run the training on (cpu or gpu)
    """
    # Prepare stream
    stream = WOSStream(transformer_model=transformer_model, transform=transform, device=device)
    stream.prepare_for_use()

    # Check for checkpoints and initialize
    model = LSTM(
        embedding_dim=utils.EMBEDDING_DIM, no_classes=stream.n_classes, device=device
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model_name = "lstm-wos-{}-ver-{}-batch".format(
        transformer_model.name, stream.version
    )
    model_path = os.path.join(PATH, model_name)
    epoch = 0
    if not os.path.exists(os.path.join(model_path, "checkpoint.pt")):
        print("Starting training from scratch...")
        os.makedirs(model_path, exist_ok=True)
    elif load_checkpoint:
        print("Resuming training from checkpoint...")
        checkpoint = torch.load(os.path.join(model_path, "checkpoint.pt"))
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        epoch = checkpoint["epoch"]
    else:
        print("Starting training from scratch...")

    criterion = nn.NLLLoss()
    losses, accuracies = [], []

    for epoch in range(epoch, epochs):
        # Initialize the loss
        running_loss = 0
        running_accuracy = 0
        # Start iterating over the dataset
        i = 0
        while stream.has_more_samples():
            # Get the batch from the stream
            if stream.n_remaining_samples() < batch_size:
                x_, y = stream.next_sample(stream.n_remaining_samples())
            else:
                x_, y = stream.next_sample(batch_size)

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
            running_accuracy += accuracy(labels=y, predictions=predictions).item()
            if i % print_every == print_every - 1:
                # Print every 10 batches
                print(
                    "[{}/{} epochs, {}/{} batches] loss: {}, accuracy: {}".format(
                        epoch + 1,
                        epochs,
                        i + 1,
                        stream.n_samples // batch_size + 1,
                        running_loss / print_every,
                        running_accuracy / print_every,
                    )
                )
                losses.append(running_loss / print_every)
                accuracies.append(running_accuracy / print_every)
                running_loss = 0
                running_accuracy = 0

            # Increment i
            i += 1

        # Save checkpoint
        print("Saving checkpoint...")
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            os.path.join(model_path, "checkpoint.pt"),
        )
        # Restart the stream
        stream.restart()

    # Save model
    print("Finished training. Saving model..")
    torch.save(model, os.path.join(model_path, "model.pt"))

    return losses, accuracies


def train_wos_stream(
    epochs=1,
    lr=0.001,
    batch_size=utils.BATCH_SIZE,
    transform=True,
    transformer_model=TransformerModel.BERT,
    eval_every=10 * utils.BATCH_SIZE,
    device="cpu",
):
    """ Trains a model using a data stream.

    Returns:

    """
    # Set the stream
    stream = WOSStream(transformer_model=transformer_model, transform=transform, device=device)
    stream.prepare_for_use()

    model_name = "lstm-wos-{}-ver-{}-batch".format(
        transformer_model.name, stream.version
    )
    model_path = os.path.join(PATH, model_name)
    # Set the model
    model = LSTMStream(
        embedding_dim=utils.EMBEDDING_DIM,
        no_classes=stream.n_classes,
        lr=lr,
        device=device,
    )

    # Set the evaluator
    evaluator = EvaluateHoldout(
        n_wait=eval_every,
        batch_size=batch_size,
        metrics=["accuracy"],
        dynamic_test_set=True,
        test_size=batch_size,
    )

    for _ in range(epochs):
        evaluator.evaluate(stream=stream, model=model, model_names=["LSTM"])

    # Save model
    print("Finished training. Saving model..")
    torch.save(model, os.path.join(model_path, "model.pt"))


if __name__ == "__main__":
    _ = train_wos_batch(epochs=1, transform=True, device='cpu')
    # train_wos_stream(epochs=1, transform=False)
