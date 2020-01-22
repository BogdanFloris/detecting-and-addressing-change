"""
Transforms a dataset using a transformer and saves it
"""
import os
import torch
from pathlib import Path
from tqdm import tqdm
from streams.loaders import load_wos
from constants.transformers import TransformerModel, Transformer


SAVE_DIR = os.path.join(Path(__file__).parents[1], "assets/datasets")


def transform_wos(version=1, transformer_model=TransformerModel.BERT, hidden_state=0):
    """ Transformers the given version of the Web of Science
    dataset using the given transformer model and saves it
    as a tuple (x, y, no_classes).

    Args:
        version (int): the WOS version
        transformer_model (TransformerModel): the transformer model to use
        hidden_state (int): which hidden state to get from the transformer
    """
    print("Transforming dataset...")
    transformer = Transformer(transformer_model)
    x, y, no_classes = load_wos(version)
    transformed_x = []

    for i in tqdm(range(len(x))):
        transformed_x.append(transformer.transform(x[i], hidden_state))

    print("Saving dataset...")
    f = open(
        os.path.join(
            SAVE_DIR,
            "wos_v_{}_transformed_{}_hidden_{}.pt".format(
                version, transformer_model.name, hidden_state
            ),
        ),
        "wb",
    )
    torch.save((transformed_x, y, no_classes), f)
    f.close()


if __name__ == "__main__":
    transform_wos(transformer_model=TransformerModel.DISTILBERT)
