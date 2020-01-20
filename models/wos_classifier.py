"""
This module implements classifiers for the Web of Science dataset
"""
import utils
import torch
from torch import nn
import torch.nn.functional as f
from skmultiflow.core import BaseSKMObject, ClassifierMixin
from streams.stream_data import WOSStream


class LSTMWrapper(BaseSKMObject, ClassifierMixin):
    def __init__(
        self,
        embedding_dim,
        no_classes,
        hidden_size=utils.HIDDEN_DIM,
        lstm_layers=utils.LSTM_LAYERS,
        device="cpu",
    ):
        self.lstm = LSTM(
            embedding_dim=embedding_dim,
            no_classes=no_classes,
            hidden_size=hidden_size,
            lstm_layers=lstm_layers,
            device=device,
        )

    def partial_fit(self, X, y, classes=None, sample_weight=None):
        pass

    def predict(self, X):
        pass

    def predict_proba(self, X):
        pass


class LSTM(nn.Module):
    """
    LSTM model for multi-class classification.
    """

    def __init__(
        self,
        embedding_dim,
        no_classes,
        hidden_size=utils.HIDDEN_DIM,
        lstm_layers=utils.LSTM_LAYERS,
        device="cpu",
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.no_classes = no_classes
        self.hidden_size = hidden_size
        self.lstm_layers = lstm_layers
        self.device = device

        # Initialize the LSTM
        self.lstm = nn.LSTM(
            input_size=self.embedding_dim,
            hidden_size=self.hidden_size,
            num_layers=self.lstm_layers,
            batch_first=True,
        )

        # Initialize the linear layer
        self.fc = nn.Linear(
            in_features=self.hidden_size, out_features=self.no_classes,
        )

    def forward(self, x, x_seq_lengths, hidden=None, cell=None):
        """ Forward pass

        Args:
            x (tensor): shape (batch_size, seq_len, embedding_dim)
            x_seq_lengths (list): list of sequence lengths for each tensor in the batch
            hidden (tensor): last hidden state
            cell (tensor): last cell state

        Returns:
            probabilities of each class
        """
        # Pack the input batch so that the padded zeros are not shown to the LSTM
        x = nn.utils.rnn.pack_padded_sequence(
            x, x_seq_lengths, batch_first=True, enforce_sorted=False
        ).to(self.device)

        # Get the LSTM output depending of whether hidden or cell have been passed
        if hidden is None or cell is None:
            x, (hidden, cell) = self.lstm(x)
        else:
            x, (hidden, cell) = self.lstm(x, (hidden, cell))

        # Unpack the packing operation
        x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)

        # Run through max pooling and linear layer
        x = self.fc(self.abs_max_pooling(x))

        # Run through softmax
        x = f.log_softmax(x, dim=1)

        return x, (hidden, cell)

    @staticmethod
    def abs_max_pooling(t, dim=1):
        """ Take absolute max pooling of tensor t, on dimension dim.

        Args:
             t (tensor): input tensor of shape (batch, seq_len, embedding_dim)
             dim (int): dimension on which to max pool
        """
        # Max over absolute value in the dimension
        _, abs_max_i = torch.max(t.abs(), dim=dim)
        # Convert indices into one hot vectors
        one_hot = f.one_hot(abs_max_i, num_classes=t.size()[dim]).transpose(dim, -1).type(torch.float)
        # Multiply original with one hot to apply mask and then sum over the dimension
        return torch.mul(t, one_hot).sum(dim=dim)


if __name__ == "__main__":
    stream_ = WOSStream()
    stream_.prepare_for_use()
    model_ = LSTM(utils.EMBEDDING_DIM, stream_.get_no_classes())
    x_, y_, seq_lengths = stream_.next_sample(utils.BATCH_SIZE)
    out, _ = model_.forward(x_, seq_lengths)
