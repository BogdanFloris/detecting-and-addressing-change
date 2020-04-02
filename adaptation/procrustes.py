""" Procrustes method to find a orthogonal linear mapping between
source and target embeddings.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.linalg import svd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from utils import constants
from adaptation.dataset import AdaptationDataset, AdaptationDatasetFullAbstracts


PATH_FIGURES = os.path.join(Path(__file__).parents[1], "assets/figures")


class Procrustes:
    def __init__(self, abstracts=False, method="average"):
        if abstracts:
            dataset = AdaptationDatasetFullAbstracts(method=method)
        else:
            dataset = AdaptationDataset(method=method)
        self.source, self.target = dataset.source, dataset.target
        self.mapping = None
        print("Creating the Procrustes mapping...")
        self.create_mapping()
        del dataset

    def create_mapping(self):
        """ Applies the procrustes method:
        https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem
        on the source and target embeddings then saves the result
        to self.mapping.
        """
        to_decompose = self.target.transpose().dot(self.source)
        # noinspection PyTupleAssignmentBalance
        u, s, v_t = svd(to_decompose, full_matrices=True)
        self.mapping = u.dot(v_t)
        assert self.mapping.shape == (constants.EMBEDDING_DIM, constants.EMBEDDING_DIM)

    def visualize_mapping(self, save_name, method="pca"):
        """ Visualizes the source, target and mapped embeddings using
        either PCA or t-SNE.

        """
        if method not in ["pca", "tsne"]:
            raise ValueError("Method should be either pca or tsne.")

        # Construct the mapping dataset
        mapping = np.matmul(self.source, self.mapping.T)

        # Construct the Data Frame
        feat_cols = ["feat" + str(i) for i in range(self.source.shape[1])]
        df = pd.DataFrame(
            np.vstack([self.source, self.target, mapping]), columns=feat_cols
        )
        df["label"] = None
        df.loc[: self.source.shape[0], "label"] = "SCIBERT"
        df.loc[self.source.shape[0] : 2 * self.source.shape[0], "label"] = "BERT"
        df.loc[2 * self.source.shape[0] :, "label"] = "Mapping SCIBERT to BERT"

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
            hue="label",
            palette=sns.color_palette("hls", 3),
            data=df,
            legend="full",
            alpha=0.3,
        )
        plt.title("Mapping visualizer {}".format(method.upper()))
        plt.tight_layout()
        plt.savefig(save_name)
        plt.show()


if __name__ == "__main__":
    proc = Procrustes()
    proc.visualize_mapping(
        save_name=os.path.join(
            PATH_FIGURES, "mapping_vis_tsne_SCIBERT_BERT_average.png"
        ),
        method="tsne",
    )
