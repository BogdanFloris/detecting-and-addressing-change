""" Training methods for the Naive Bayes model on the Web of Science dataset.
"""
import os
from pathlib import Path
from joblib import dump
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import utils
from constants.transformers import TransformerModel
from streams.stream_data import WOSStream
from utils.metrics import get_metrics
import warnings
warnings.filterwarnings("ignore")


PATH = os.path.join(Path(__file__).parents[1], "assets/models")
if not os.path.isdir(PATH):
    os.makedirs(PATH)


def train_nb_wos_holdout(
    epochs=1,
    batch_size=utils.BATCH_SIZE,
    transform=True,
    transformer_model=TransformerModel.BERT,
    print_every=10,
    device="cpu",
):
    """ Trains the Naive Bayes model on the Web of Science dataset.

    Args:
        epochs (int): number of times the stream is run
        batch_size (int): the batch size
        transform (bool): transform the dataset or not
        transformer_model (TransformerModel): the transformer model to use
        print_every (int): print stats parameter
        device (string): the device to run the training on (cpu or gpu)

    """
    # Prepare the stream
    stream = WOSStream(
        transformer_model=transformer_model, transform=transform, device=device
    )
    stream.prepare_for_use()

    # Define model
    model = GaussianNB()
    model_name = "naive-bayes-wos-{}-ver-{}-holdout".format(
        transformer_model.name, stream.version
    )
    model_path = os.path.join(PATH, model_name)
    os.makedirs(model_path, exist_ok=True)
    all_labels = np.arange(stream.n_classes)

    print("Starting training...")
    train_accuracies, test_metrics_list = [], []

    for epoch in range(epochs):
        # Initialize the running loss and accuracy
        running_accuracy = 0.0
        # Start iterating over the dataset
        i = 0
        while stream.has_more_samples():
            # Get the batch from the stream
            if stream.n_remaining_samples() < batch_size:
                x_, y = stream.next_sample(stream.n_remaining_samples())
            else:
                x_, y = stream.next_sample(batch_size)

            # Unpack x_ (we do not need the sequence lengths for NB)
            x = x_[0].numpy()
            # Pad and reshape
            x = np.pad(
                x, ((0, 0), (0, utils.MAX_SEQ_LEN - x.shape[1]), (0, 0))
            ).reshape(batch_size, -1)

            # Partial fit the model
            model.partial_fit(x, y, classes=all_labels)

            # Update running accuracy
            running_accuracy += accuracy_score(y, model.predict(x))

            # Print statistics
            if i % print_every == print_every - 1 or not stream.has_more_samples():
                # Evaluate the model on the test set
                x_test_, y_test = stream.get_test_set()
                x_test = x_test_[0].numpy()
                x_test = np.pad(
                    x_test, ((0, 0), (0, utils.MAX_SEQ_LEN - x_test.shape[1]), (0, 0))
                ).reshape(x_test.shape[0], -1)
                y_pred = model.predict(x_test)
                test_metrics = get_metrics(y_pred, y_test, no_labels=stream.n_classes)

                denominator = (
                    print_every
                    if stream.has_more_samples()
                    else (stream.n_samples // batch_size + 1) % print_every
                )
                accuracy = running_accuracy / denominator

                # Print every 10 batches
                print(
                    "[{}/{} epochs, {}/{} batches] train accuracy: {:.4f}, "
                    "test (accuracy: {:.4f}, precision: {:.4f}, "
                    "recall: {:.4f}, f1: {:.4f})".format(
                        epoch + 1,
                        epochs,
                        i + 1,
                        stream.n_samples // batch_size + 1,
                        accuracy,
                        test_metrics["accuracy"],
                        test_metrics["precision"],
                        test_metrics["recall"],
                        test_metrics["macro_f1"],
                    )
                )
                train_accuracies.append(accuracy)
                test_metrics_list.append(test_metrics)
                running_accuracy = 0

            # Increment i
            i += 1

        stream.restart()

    # Save model
    print("Finished training. Saving model..")
    dump(model, os.path.join(model_path, "model.joblib"))
    print("Done!")

    return train_accuracies, test_metrics_list


if __name__ == "__main__":
    _ = train_nb_wos_holdout(epochs=0, transform=False, print_every=1, device="cpu")
