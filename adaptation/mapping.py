""" Procrustes method to find a orthogonal linear mapping between
source and target embeddings.
"""
import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from abc import ABC, abstractmethod
from pathlib import Path
from torch import nn, optim
from torch.utils import data
from scipy.linalg import svd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from utils import constants
from models.adaptation_models import MLP
from adaptation.dataset import AdaptationDataset, AdaptationDatasetFullAbstracts


PATH_FIGURES = os.path.join(Path(__file__).parents[1], "assets/figures")
PATH_MODELS = os.path.join(Path(__file__).parents[1], "assets/models")


class Mapping(ABC):
    def __init__(self, abstracts=False, method="average", x_most_common=5000):
        if abstracts:
            dataset = AdaptationDatasetFullAbstracts(method=method)
        else:
            dataset = AdaptationDataset(method=method, x_most_common=x_most_common)
        self.source, self.target = dataset.source, dataset.target
        self.mapping = None

    @abstractmethod
    def create_mapping(self):
        raise NotImplementedError()

    @staticmethod
    def mse_loss(source, target):
        return (np.square(source - target)).mean()

    def visualize_mapping(self, save_name, method="pca"):
        """ Visualizes the source, target and mapped embeddings using
        either PCA or t-SNE.

        """
        if method not in ["pca", "tsne"]:
            raise ValueError("Method should be either pca or tsne.")

        # Construct the mapping dataset
        if type(self.mapping) == np.ndarray:
            mapping = np.matmul(self.source, self.mapping.T)
        else:
            mapping = self.mapping(self.source).detach().numpy()

        # Construct the Data Frame
        feat_cols = ["feat" + str(i) for i in range(self.source.shape[1])]
        df = pd.DataFrame(
            np.vstack([self.source, self.target, mapping]), columns=feat_cols
        )
        df["embedding"] = None
        df.loc[: self.source.shape[0], "embedding"] = "SCIBERT"
        df.loc[self.source.shape[0] : 2 * self.source.shape[0], "embedding"] = "BERT"
        df.loc[2 * self.source.shape[0] :, "embedding"] = "Mapping SCIBERT to BERT"

        if method == "pca":
            print("Running PCA")
            dim_reduction = PCA(n_components=2)
        else:
            print("Running TSNE")
            dim_reduction = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)

        pca_result = dim_reduction.fit_transform(df[feat_cols].values)
        df["dim-one"] = pca_result[:, 0]
        df["dim-two"] = pca_result[:, 1]

        sns.set(style="darkgrid")
        plt.figure(figsize=(16, 10))
        sns.scatterplot(
            x="dim-one",
            y="dim-two",
            hue="embedding",
            palette=sns.color_palette("hls", 3),
            data=df,
            legend="full",
            alpha=0.3,
        )
        plt.title("Mapping visualizer {}".format(method.upper()))
        plt.tight_layout()
        plt.savefig(save_name)
        plt.show()


class Procrustes(Mapping):
    def __init__(self, abstracts=False, method="average", x_most_common=5000):
        super().__init__(abstracts, method, x_most_common)
        print("Creating Procrustes mapping")
        self.create_mapping()

    def create_mapping(self, iterations=10):
        """ Applies the procrustes method:
        https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem
        on the source and target embeddings then saves the result
        to self.mapping.
        """
        print(
            "MSE loss before mapping: {}".format(
                self.mse_loss(self.source, self.target)
            )
        )
        to_decompose = self.target.transpose().dot(self.source)
        # noinspection PyTupleAssignmentBalance
        u, s, v_t = svd(to_decompose, full_matrices=True)
        self.mapping = u.dot(v_t)
        assert self.mapping.shape == (constants.EMBEDDING_DIM, constants.EMBEDDING_DIM)
        print(
            "MSE loss after mapping: {}".format(
                self.mse_loss(self.map(self.source), self.target)
            )
        )

    def map(self, inputs):
        return inputs.dot(self.mapping.T)


class MLPMapping(Mapping):
    def __init__(self, abstracts=False, method="average", x_most_common=10000):
        super().__init__(abstracts, method, x_most_common)
        self.name = "mlp_mapping_{}_{}".format(method, x_most_common)
        if not os.path.isdir(os.path.join(PATH_MODELS, self.name)):
            os.makedirs(os.path.join(PATH_MODELS, self.name))
        self.model_path = os.path.join(os.path.join(PATH_MODELS, self.name), "model.pt")
        self.mapping = MLP(embedding_dim=self.source.shape[1])
        if not os.path.exists(self.model_path):
            print("Training MLP mapping")
            self.create_mapping()
        else:
            print("Loading MLP mapping")
            self.mapping.load_state_dict(torch.load(self.model_path))

    def create_mapping(self, epochs=10, lr=0.001, batch_size=50):
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.mapping.parameters(), lr=lr)
        dataset = data.TensorDataset(
            torch.tensor(self.source, dtype=torch.float),
            torch.tensor(self.target, dtype=torch.float),
        )
        data_loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        print(
            "MSE loss before mapping: {}".format(
                self.mse_loss(self.source, self.target)
            )
        )
        for epoch in range(epochs):
            for i, (x, y) in enumerate(data_loader):
                optimizer.zero_grad()
                predictions = self.mapping(x)
                loss = criterion(predictions, y)
                loss.backward()
                optimizer.step()
                if i % 10 == 0:
                    with torch.no_grad():
                        train_loss = self.mse_loss(
                            self.mapping(self.source).detach().numpy(), self.target
                        )
                    print(
                        "[Epoch {}/{}, Iter {}/{}] train loss: {}, test loss: {}".format(
                            epoch + 1,
                            epochs,
                            i + 1,
                            len(data_loader),
                            loss.item(),
                            train_loss,
                        )
                    )

        torch.save(self.mapping.state_dict(), self.model_path)


if __name__ == "__main__":
    m = MLPMapping()
    m.visualize_mapping(
        save_name=os.path.join(
            PATH_FIGURES, "mlp_mapping_vis_tsne_SCIBERT_BERT_average.png"
        ),
        method="tsne",
    )
