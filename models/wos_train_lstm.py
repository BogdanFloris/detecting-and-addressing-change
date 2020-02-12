""" Training methods for the LSTM model on the Web of Science dataset.
"""
import os
import torch
import utils
from pathlib import Path
from torch import nn, optim
from sklearn.metrics import accuracy_score
from streams.stream_data import WOSStream
from models.wos_classifier import LSTM
from constants.transformers import TransformerModel
from utils.metrics import get_metrics


PATH = os.path.join(Path(__file__).parents[1], "assets/models")
if not os.path.isdir(PATH):
    os.makedirs(PATH)


def train_lstm_wos_holdout(
    epochs=1,
    lr=0.001,
    batch_size=utils.BATCH_SIZE,
    transform=True,
    transformer_model=TransformerModel.BERT,
    print_every=10,
    load_checkpoint=False,
    device="cpu",
):
    """ Trains the LSTM model on the Web of Science dataset.

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
    stream = WOSStream(
        transformer_model=transformer_model, transform=transform, device=device
    )
    stream.prepare_for_use()

    # Check for checkpoints and initialize
    model = LSTM(
        embedding_dim=utils.EMBEDDING_DIM, no_classes=stream.n_classes, device=device
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model_name = "lstm-wos-{}-ver-{}-holdout".format(
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
    losses, train_accuracies, test_metrics_list = [], [], []

    for epoch in range(epoch, epochs):
        # Initialize the running_loss and accuracy
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
            running_accuracy += accuracy_score(y.cpu(), predictions.argmax(dim=1).cpu())
            if i % print_every == print_every - 1 or not stream.has_more_samples():
                # Evaluate the model on the test set
                x_test_, y_test = stream.get_test_set()
                x_test, seq_len_test = x_test_
                x_test = x_test.to(device)
                y_test = torch.from_numpy(y_test).to(device)
                seq_len_test = torch.tensor(seq_len_test).to(device)

                with torch.no_grad():
                    test_pred, _ = model((x_test, seq_len_test))
                    test_metrics = get_metrics(
                        labels=y_test.cpu(),
                        predictions=test_pred.cpu(),
                        no_labels=stream.n_classes,
                    )

                denominator = (
                    print_every
                    if stream.has_more_samples()
                    else (stream.n_samples // batch_size + 1) % print_every
                )
                train_loss = running_loss / denominator
                train_acc = running_accuracy / denominator

                # Print every 10 batches
                print(
                    "[{}/{} epochs, {}/{} batches] train loss: {:.4f}, "
                    "train accuracy: {:.4f}, test (accuracy: {:.4f}, precision: {:.4f}, "
                    "recall: {:.4f}, f1: {:.4f})".format(
                        epoch + 1,
                        epochs,
                        i + 1,
                        stream.n_samples // batch_size + 1,
                        train_loss,
                        train_acc,
                        test_metrics["accuracy"],
                        test_metrics["precision"],
                        test_metrics["recall"],
                        test_metrics["macro_f1"],
                    )
                )
                losses.append(train_loss)
                train_accuracies.append(train_acc)
                test_metrics_list.append(test_metrics)
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
    torch.save(model.state_dict(), os.path.join(model_path, "model.pt"))
    print("Done!")

    return losses, train_accuracies, test_metrics_list


if __name__ == "__main__":
    _ = train_lstm_wos_holdout(epochs=1, transform=False, print_every=1, device="cpu")
