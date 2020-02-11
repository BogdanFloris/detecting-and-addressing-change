"""
This file contains streams classes that generate different drift_datasets
over time. The streams are based on the sk-multiflow framework.
"""
import os
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.model_selection import train_test_split
from skmultiflow.data.base_stream import Stream
from constants.transformers import Transformer, TransformerModel
from streams.loaders import load_wos

import utils


PATH = os.path.join(Path(__file__).parents[1], "assets/datasets")
TRANSFORMED_DATASETS = [
    os.path.join(PATH, "wos_v_1_transformed_BERT_hidden_0.pt"),
    os.path.join(PATH, "wos_v_1_transformed_SCIBERT_hidden_0.pt"),
    os.path.join(PATH, "wos_v_1_transformed_DISTILBERT_hidden_0.pt"),
]


class WOSStream(Stream):
    """ Class that abstracts the Web of Science dataset into a streaming dataset.
    There are 3 versions of the dataset (1, 2, 3), each having
    increasingly more samples and targets.

    When calling `next_sample` this class also transforms the text
    to contextualized embeddings based on the given transformer.

    Args:
        version (int): the version of the dataset (1, 2, 3)
        transformer_model (TransformerModel): the transformer model to use
        transform (bool): whether to transform the dataset while streaming,
            or use an already transformed dataset
        test_split (bool): whether to train test split the data or not
        shuffle (bool): whether or not to shuffle the test set every time the stream is restarted
        device (str): cuda or cpu
    """

    def __init__(
        self,
        version=1,
        transformer_model=TransformerModel.BERT,
        transform=True,
        test_split=True,
        shuffle=False,
        device="cpu",
    ):
        super().__init__()
        self.version = version
        self.X = None
        self.y = None
        self.X_test = None
        self.y_test = None
        self.name = "WOS-{}".format(transformer_model.name)
        self.n_targets = 1
        self.n_features = utils.EMBEDDING_DIM
        self.current_seq_lengths = None
        self.transform = transform
        self.test_split = test_split
        self.shuffle = shuffle
        if test_split:
            self.test_samples = None
            self.test_sequence_lengths = None

        if transformer_model == TransformerModel.BERT:
            self.dataset_idx = 0
        elif transformer_model == TransformerModel.SCIBERT:
            self.dataset_idx = 1
        else:
            self.dataset_idx = 2

        if transform:
            self.transformer = Transformer(transformer_model, device=device)

    def prepare_for_use(self):
        """Prepares the stream for use by initializing
        the X and y variables from the files.
        """
        if self.transform:
            print("Preparing non-transformed dataset...")
            self.X, self.y, self.n_classes = load_wos(version=self.version)
        else:
            print("Preparing transformed dataset...")
            self.X, self.y, self.n_classes = torch.load(
                TRANSFORMED_DATASETS[self.dataset_idx]
            )
        if self.test_split:
            self.X, self.X_test, self.y, self.y_test = train_test_split(
                self.X, self.y, test_size=0.2
            )
        self.n_samples = len(self.y)
        self.sample_idx = 0
        self.current_sample_x = None
        self.current_sample_y = None
        self.current_seq_lengths = None

    def restart(self):
        """Restarts the stream.
        """
        if self.test_split and self.shuffle:
            self.X, self.X_test, self.y, self.y_test = train_test_split(
                self.X, self.y, test_size=0.2
            )
            self.test_samples = None
            self.test_sequence_lengths = None
        self.n_samples = len(self.y)
        self.sample_idx = 0
        self.current_sample_x = None
        self.current_sample_y = None
        self.current_seq_lengths = None

    def next_sample(self, batch_size=1):
        """Returns the next batch_size samples if the number of samples
        is available, and otherwise returns None, None.

        Args:
            batch_size (int): the number of samples to return

        Returns:
            tuple or list of tuples containing the samples
        """
        self.sample_idx += batch_size
        try:
            # Get the text and transform it to embeddings
            self.current_sample_x = self.X[
                self.sample_idx - batch_size : self.sample_idx
            ]
            self.current_seq_lengths = []
            for i in range(len(self.current_sample_x)):
                # Transform to embeddings
                if self.transform:
                    self.current_sample_x[i] = self.transformer.transform(
                        self.current_sample_x[i]
                    )
                # Squeeze tensor
                self.current_sample_x[i] = self.current_sample_x[i].squeeze()

                # Save the sequence length
                self.current_seq_lengths.append(self.current_sample_x[i].shape[0])

            # Pad and stack sequence
            self.current_sample_x = nn.utils.rnn.pad_sequence(
                self.current_sample_x, batch_first=True
            )

            # Get the y target
            self.current_sample_y = self.y[
                self.sample_idx - batch_size : self.sample_idx
            ]
        except IndexError:
            self.current_sample_x = None
            self.current_sample_y = None
        return (self.current_sample_x, self.current_seq_lengths), self.current_sample_y

    def get_test_set(self):
        """ Returns the test set if the class variables test split is True,
        otherwise raises an error.
        """
        # If test split is not set, raise error
        if not self.test_split:
            raise ValueError("Test set is not available for this stream.")

        # If test samples have not been preprocessed yet, preprocess
        if self.test_samples is None:
            self.test_samples = self.X_test
            self.test_sequence_lengths = []
            for i in range(len(self.test_samples)):
                # Transform to embeddings
                if self.transform:
                    self.test_samples[i] = self.transformer.transform(
                        self.test_samples[i]
                    )
                # Squeeze the tensor
                self.test_samples[i] = self.test_samples[i].squeeze()
                # Save the sequence length
                self.test_sequence_lengths.append(self.test_samples[i].shape[0])

            # Pad and stack test set
            self.test_samples = nn.utils.rnn.pad_sequence(
                self.test_samples, batch_first=True
            )

        return (self.test_samples, self.test_sequence_lengths), self.y_test

    def n_remaining_samples(self):
        """Returns the number of samples left in the stream.

        Returns:
            the number of samples left in the stream
        """
        return self.n_samples - self.sample_idx

    def has_more_samples(self):
        """Returns true if there are more samples in the stream, and False otherwise.

        Returns:
            True, if more samples are available, False otherwise
        """
        return self.sample_idx < self.n_samples

    def get_no_classes(self):
        return self.n_classes


if __name__ == "__main__":
    wos = WOSStream(transformer_model=TransformerModel.SCIBERT, transform=False)
    wos.prepare_for_use()
    x, y = wos.next_sample(8)
    print(x[0].shape)
