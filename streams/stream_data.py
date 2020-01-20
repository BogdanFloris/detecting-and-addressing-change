"""
This file contains streams classes that generate different drift_datasets
over time. The streams are based on the sk-multiflow framework.
"""
import torch.nn as nn
from skmultiflow.data.base_stream import Stream
from constants.transformers import Transformer, TransformerModel
from streams.loaders import load_wos


class WOSStream(Stream):
    """ Class that abstracts the Web of Science dataset into a streaming dataset.
    There are 3 versions of the dataset (1, 2, 3), each having
    increasingly more samples and targets.

    When calling `next_sample` this class also transforms the text
    to contextualized embeddings based on the given transformer.
    """

    def __init__(self, version=1, transformer_model=TransformerModel.BERT):
        super().__init__()
        self.version = version
        self.X = None
        self.y = None
        self.n_samples = None
        self.no_classes = None
        self.current_seq_lengths = None

        self.transformer = Transformer(transformer_model)

    def prepare_for_use(self):
        """Prepares the stream for use by initializing
        the X and y variables from the files.
        """
        self.X, self.y, self.no_classes = load_wos(version=self.version)
        self.n_samples = len(self.y)
        self.sample_idx = 0
        self.current_sample_x = None
        self.current_sample_y = None
        self.current_seq_lengths = None

    def restart(self):
        """Restarts the stream.
        """
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
                self.current_sample_x[i] = self.transformer.transform(
                    self.current_sample_x[i]
                ).squeeze()

                # Save the sequence length
                self.current_seq_lengths.append(self.current_sample_x[i].shape[0])

            # Pad and stack sequence
            self.current_sample_x = nn.utils.rnn.pad_sequence(self.current_sample_x, batch_first=True)

            # Get the y target
            self.current_sample_y = self.y[
                self.sample_idx - batch_size : self.sample_idx
            ]
        except IndexError:
            self.current_sample_x = None
            self.current_sample_y = None
        return self.current_sample_x, self.current_sample_y, self.current_seq_lengths

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
        return self.no_classes


if __name__ == "__main__":
    wos = WOSStream(transformer_model=TransformerModel.SCIBERT)
    wos.prepare_for_use()
    x, y, _ = wos.next_sample(8)
    print(x.shape)
