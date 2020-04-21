import os
import numpy as np
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

from constants.transformers import Transformer, TransformerModel
from streams.loaders import load_wos
from streams.stream_data import WOSStream
from utils.formatting import clean_text
from utils import constants


PATH = os.path.join(Path(__file__).parents[1], "assets/datasets")


class AdaptationDatasetFullAbstracts:
    """ This is used to store and create the dataset
    that will be used in the adaptation experiments.
    It uses the full abstracts averages, instead of the most common words.

    Args:
        version (int): the version of the dataset (1, 2, 3)
        transformer_model_source (TransformerModel): the transformer model
            used as a source embedding
        transformer_model_target (TransformerModel): the transformer model
            used as a target embedding
        method (str): what method to use to pick the embedding when a word is tokenized
            into more than one token or the number of tokens is different between
            the source and the target embeddings. Can pick two options
             - average: averages all the token embeddings
             - max: picks the maximum over the token axis

    """

    def __init__(
        self,
        version=1,
        transformer_model_source=TransformerModel.SCIBERT,
        transformer_model_target=TransformerModel.BERT,
        method="average",
    ):
        # Initialize the name of the dataset
        self.name = "ad_abst_v_{}_{}_{}_{}.npy".format(
            version,
            transformer_model_source.name,
            transformer_model_target.name,
            method,
        )
        # Initialize method
        if method in ["average", "max"]:
            self.method = method
        else:
            raise ValueError("Must pick method average or max.")

        if os.path.isfile(os.path.join(PATH, self.name)):
            # Save already exists, so load
            print("Loading the adaptation dataset...")
            self.source, self.target = np.load(os.path.join(PATH, self.name))
        else:
            # Create the dataset
            print("Creating the full abstract adaptation dataset...")

            wos_source = WOSStream(
                version=version,
                transformer_model=transformer_model_source,
                transform=False,
                test_split=False,
            )
            wos_source.prepare_for_use()
            wos_target = WOSStream(
                version=version,
                transformer_model=transformer_model_target,
                transform=False,
                test_split=False,
            )
            wos_target.prepare_for_use()

            self.source = np.zeros(
                shape=(wos_source.n_samples, constants.EMBEDDING_DIM), dtype=np.float
            )
            self.target = np.zeros(
                shape=(wos_target.n_samples, constants.EMBEDDING_DIM), dtype=np.float
            )

            for i in tqdm(range(wos_source.n_samples)):
                (x_source, _), _ = wos_source.next_sample()
                (x_target, _), _ = wos_target.next_sample()
                if self.method == "average":
                    self.source[i] = np.mean(x_source.squeeze().numpy(), axis=0)
                    self.target[i] = np.mean(x_target.squeeze().numpy(), axis=0)
                elif self.method == "max":
                    self.source[i] = np.mean(x_source.squeeze().numpy(), axis=0)
                    self.target[i] = np.mean(x_target.squeeze().numpy(), axis=0)

            # Save dataset
            np.save(os.path.join(PATH, self.name), (self.source, self.target))


class AdaptationDataset:
    """ This is used to store and create the dataset
    that will be used in the adaptation experiments.

    Args:
        version (int): the version of the dataset (1, 2, 3)
        x_most_common (int): how many words to use in the dataset
        transformer_model_source (TransformerModel): the transformer model
            used as a source embedding
        transformer_model_target (TransformerModel): the transformer model
            used as a target embedding
        method (str): what method to use to pick the embedding when a word is tokenized
            into more than one token or the number of tokens is different between
            the source and the target embeddings. Can pick two options
             - first: picks the first token embedding
             - average: averages all the token embeddings
             - max: picks the maximum over the token axis
        device (str): cuda or cpu

    """

    def __init__(
        self,
        version=1,
        x_most_common=5000,
        transformer_model_source=TransformerModel.SCIBERT,
        transformer_model_target=TransformerModel.BERT,
        method="average",
        device="cpu",
    ):
        # Initialize the name of the dataset
        self.name = "ad_v_{}_{}_{}_{}_{}.npy".format(
            version,
            transformer_model_source.name,
            transformer_model_target.name,
            x_most_common,
            method,
        )
        # Initialize method
        if method in ["first", "average", "max"]:
            self.method = method
        else:
            raise ValueError("Must pick method first, average, or max.")

        if os.path.isfile(os.path.join(PATH, self.name)):
            # Save already exists, so load
            print("Loading the adaptation dataset...")
            self.source, self.target = np.load(os.path.join(PATH, self.name))
        else:
            # Create the dataset
            print("Creating the adaptation dataset...")
            self.x_most_common = x_most_common

            # Initialize the transformers
            self.transformer_source = Transformer(
                model=transformer_model_source, device=device
            )
            self.transformer_target = Transformer(
                model=transformer_model_target, device=device
            )

            # Initialize the Web of Science abstracts
            self.abstracts, _, _ = load_wos(version=version)

            # Initialize the word to count dictionary,
            # most common words lists, and the actual dataset
            self.word_to_count = defaultdict(int)
            self.most_common_words = []
            self.source = np.zeros(
                shape=(x_most_common, constants.EMBEDDING_DIM), dtype=np.float
            )
            self.target = np.zeros(
                shape=(x_most_common, constants.EMBEDDING_DIM), dtype=np.float
            )

            # Create the dataset
            self.create()

            # Save dataset
            np.save(os.path.join(PATH, self.name), (self.source, self.target))

    def create(self):
        """ Creates the dataset.
        """
        # Create the word to count dictionary
        self.create_word_to_count()

        # Get the x most common words
        self.get_most_common_words()

        # Transform the most common words to create the dataset
        self.transform_words()

    def create_word_to_count(self):
        """ Creates the word to count dictionary.
        """
        print("Creating the word to count dictionary...")
        for abstract in self.abstracts:
            abstract = clean_text(abstract, remove_punctuation=True)
            words = abstract.split(" ")
            for word in words:
                self.word_to_count[word] += 1

    def get_most_common_words(self):
        """ Get the x most common words
        """
        print("Getting the most common {} words...".format(self.x_most_common))
        self.most_common_words = [
            word
            for word, _ in sorted(
                self.word_to_count.items(), key=lambda item: item[1], reverse=True
            )
        ][: self.x_most_common]

    def transform_words(self):
        """ Transforms the most common words using both the
        source and target transformers to create the dataset.
        """
        print("Transforming words to dataset...")
        for i, word in tqdm(enumerate(self.most_common_words)):
            source_embeddings = self.transformer_source.transform(word)[0].numpy()
            target_embeddings = self.transformer_target.transform(word)[0].numpy()
            if self.method == "first":
                self.source[i] = source_embeddings[0]
                self.target[i] = target_embeddings[0]
            elif self.method == "average":
                self.source[i] = np.mean(source_embeddings, axis=0)
                self.target[i] = np.mean(target_embeddings, axis=0)
            elif self.method == "max":
                self.source[i] = np.max(source_embeddings, axis=0)
                self.target[i] = np.max(target_embeddings, axis=0)


if __name__ == "__main__":
    ad = AdaptationDataset(x_most_common=10000)
