""" Experiments with the Procrustes mapping.
"""
import os
import sys
import pickle
from pathlib import Path
import torch
import utils
from skmultiflow.drift_detection import DDM
from adaptation.mapping import Procrustes
from adaptation.stream import run_stream_with_mapping
from models.wos_classifier import LSTM
from streams.stream_data import WOSStream
from streams.run_stream_supervised import run_stream_lstm
from constants.transformers import TransformerModel


PATH_MODELS = os.path.join(Path(__file__).parents[1], "assets/models")
PATH_RESULTS = os.path.join(Path(__file__).parents[1], "assets/results")
LSTM_MODELS = [
    os.path.join(PATH_MODELS, "lstm-wos-BERT-ver-1-holdout/model.pt"),
]


def procrustes_experiment(
    save_name,
    lstm_model_idx=0,
    transformer_model_trained=TransformerModel.BERT,
    transformer_model_untrained=TransformerModel.SCIBERT,
    method="average",
    batch_size=1,
    transform=True,
    print_every=1,
    device="cpu",
):
    """
    Runs an adaptation experiments using the Procrustes linear mapping.

    Args:
        save_name (str): name of the file where the function saves the result
        lstm_model_idx (int): the index of the LSTM model (from the available ones)
        transformer_model_trained (TransformerModel): the embeddings on which the model was trained
        transformer_model_untrained (TransformerModel): the embeddings against which the model is compared
        method (str): method parameter used for picking the adaptation dataset
        batch_size (int): the batch size for the stream
        transform (bool): whether to transform the text or used pre transformed one
        print_every (int): how often to print
        device (str): cpu or cuda

    Returns:
        a dictionary with the results

    """
    # Add method to save name
    save_name += "_" + method
    # Initialize the stream that the model was trained on
    stream_trained = WOSStream(
        transformer_model=transformer_model_trained,
        transform=transform,
        test_split=False,
        device=device,
    )
    stream_trained.prepare_for_use()
    # Initialize the stream with other embeddings, to add drift
    stream_untrained = WOSStream(
        transformer_model=transformer_model_untrained,
        transform=transform,
        test_split=False,
        device=device,
    )
    stream_untrained.prepare_for_use()
    # Initialize the adaptation dataset
    linear_mapping = Procrustes(method=method, x_most_common=10000)
    # Load the LSTM model
    model = LSTM(
        embedding_dim=utils.EMBEDDING_DIM, no_classes=stream_trained.n_classes
    ).to(device)
    model.load_state_dict(
        torch.load(LSTM_MODELS[lstm_model_idx], map_location=device), strict=False
    )
    model.eval()

    # Initialize the drift detector
    drift_detector = DDM()

    # Run streams
    print("Running trained stream...")
    trained_accuracies = run_stream_lstm(
        stream_trained,
        model,
        drift_detector,
        batch_size=batch_size,
        print_every=print_every,
        warm_start=sys.maxsize,
        device=device,
    )
    print("Running untrained stream...")
    untrained_accuracies = run_stream_lstm(
        stream_untrained,
        model,
        drift_detector,
        batch_size=batch_size,
        print_every=print_every,
        warm_start=sys.maxsize,
        device=device,
    )

    # Run the stream with a mapping
    stream_untrained.restart()
    print("Running mapping stream...")
    mapping_accuracies = run_stream_with_mapping(
        stream_untrained,
        model,
        linear_mapping,
        batch_size=batch_size,
        print_every=print_every,
    )

    # Save the results
    to_save = {
        "trained_accuracies": trained_accuracies,
        "untrained_accuracies": untrained_accuracies,
        "mapping_accuracies": mapping_accuracies,
    }

    with open(os.path.join(PATH_RESULTS, save_name + ".pkl"), "wb") as f:
        pickle.dump(to_save, f)

    return to_save


if __name__ == "__main__":
    procrustes_experiment(
        "procrustes_lstm_wos_1_BERT_SCIBERT_10000_words",
        method="average",
        batch_size=32,
        transform=False,
    )
