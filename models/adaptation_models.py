""" Deep learning models used to map from one embedding space to another.
"""
import torch
import numpy as np
from torch import nn
from utils import constants


class MLP(nn.Module):
    def __init__(self, embedding_dim=constants.EMBEDDING_DIM):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.fc1 = nn.Linear(in_features=embedding_dim, out_features=embedding_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(in_features=embedding_dim, out_features=embedding_dim)

    def forward(self, inputs):
        """ Forward pass of the MLP

        Args:
            inputs (tensor): shape (batch_size, seq_len, embedding_dim)
                or (dictionary_size, embedding_dim)

        Returns:
            tensor with the same shape of input
        """
        if type(inputs) == np.ndarray:
            inputs = torch.tensor(inputs, dtype=torch.float)
        assert inputs.shape[-1] == self.embedding_dim
        out = self.fc2(self.relu(self.fc1(inputs)))
        assert out.shape == inputs.shape
        return out


if __name__ == "__main__":
    model = MLP()
    model.eval()
    x1 = np.random.random((32, 500, 768))
    x2 = np.random.random((5000, 768))
    print(model(x1).detach().numpy().shape)
    print(model(x2).detach().numpy().shape)
