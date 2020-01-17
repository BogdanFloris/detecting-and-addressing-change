"""
Loads different drift_datasets from their specific files.
Requires that the user has the folder assets/datasets with the dataset
that he wishes to use in the root folder.
"""
import os
from pathlib import Path
import numpy as np
from utils.formatting import clean_text


DATASETS = os.path.join(Path(__file__).parents[1], "assets/datasets")


def load_wos(version=1):
    """Loads the Web Of Science dataset into memory.
    There are three version of the dataset, each with
    more samples and more targets.

    Args:
        version (int): the version of the dataset (1, 2, 3)

    Returns:
        the inputs X and the targets y
    """
    if version == 1:
        wos_version = "5736"
        no_classes = 11
    elif version == 2:
        wos_version = "11967"
        no_classes = 35
    elif version == 3:
        wos_version = "46985"
        no_classes = 134
    else:
        raise ValueError("Incorrect version")
    x_fname = os.path.join(DATASETS, f"wos/WOS{wos_version}/X.txt")
    y_fname = os.path.join(DATASETS, f"wos/WOS{wos_version}/Y.txt")
    # Open and clean X
    try:
        with open(x_fname, encoding="utf-8") as f:
            content_x = f.readlines()
            content_x = [clean_text(x) for x in content_x]
    except FileNotFoundError as e:
        print(e)
    # Open and clean Y
    try:
        with open(y_fname, encoding="utf-8") as f:
            content_y = f.readlines()
            content_y = [x.strip() for x in content_y]
            label = np.array(content_y, dtype=int).transpose()
    except FileNotFoundError as e:
        print(e)

    return content_x, label, no_classes
