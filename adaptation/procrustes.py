""" Procrustes method to find a orthogonal linear mapping between
source and target embeddings.
"""
import os
from pathlib import Path
from scipy.linalg import svd
from utils import constants
from adaptation.dataset import AdaptationDataset


PATH = os.path.join(Path(__file__).parents[1], "assets/datasets")


class Procrustes:
    def __init__(self, method="first"):
        dataset = AdaptationDataset(method=method)
        self.source, self.target = dataset.source, dataset.target
        self.mapping = None
        print("Creating the Procrustes mapping...")
        self.procrustes()
        del dataset

    def procrustes(self):
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


if __name__ == "__main__":
    proc = Procrustes()
