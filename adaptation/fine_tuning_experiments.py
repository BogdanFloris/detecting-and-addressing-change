""" Experiments with fine tuning a model.
"""
import os
import sys

import torch
import utils
import pickle
from pathlib import Path
from joblib import load
from skmultiflow.drift_detection import DDM
from streams.stream_data import WOSStream
from streams.run_stream_supervised import run_stream_nb, run_stream_lstm
from adaptation.fine_tuning import fine_tune
from models.wos_classifier import LSTM
from constants.transformers import TransformerModel


PATH_MODELS = os.path.join(Path(__file__).parents[1], "assets/models")
PATH_RESULTS = os.path.join(Path(__file__).parents[1], "assets/results")
LSTM_MODELS = [
    os.path.join(PATH_MODELS, "lstm-wos-BERT-ver-1-holdout/model.pt"),
]
NB_MODELS = [
    os.path.join(PATH_MODELS, "naive-bayes-wos-BERT-ver-1-holdout/model.joblib"),
]


def fine_tuning_different_embeddings(
    save_name,
    lstm_model_idx=None,
    nb_model_idx=None,
    transformer_model_trained=None,
    transformer_model_untrained=None,
    batch_size=1,
    no_batches=50,
    transform=False,
    print_every=1,
    device="cpu",
):
    """ Performs a fine tuning experiment with 3 streams on the same model.
    The first stream is the one with embeddings on which the model was trained on.
    The second stream is the one with embeddings that are different from the ones
    on which the model was trained on.
    The third stream is the fine tuned one on the stream with embeddings different
    from the ones on which the model was trained on.
    The goal of the experiment is to see if a small abrupt concept drift can be corrected
    by fine tuning the model on the new embeddings (training it on a few batches).

    Args:
        save_name (str): name of the file where the function saves the result
        lstm_model_idx (int): the index of the LSTM model (from the available ones)
        nb_model_idx (int): the index of the Naive Bayes model (from the available ones)
        transformer_model_trained (TransformerModel): the embeddings on which the model was trained
        transformer_model_untrained (TransformerModel): the embeddings against which the model is compared
        batch_size (int): the batch size for the stream
        no_batches (int): the number of batches to fine tune on
        transform (bool): whether to transform the text or used pre transformed one
        print_every (int): how often to print
        device (str): cpu or cuda

    Returns:
        a dictionary with the results

    """
    # Initialize the stream that the model was trained on
    stream_trained = WOSStream(
        transformer_model=transformer_model_trained,
        transform=transform,
        test_split=False,
        device=device,
    )
    stream_trained.prepare_for_use()
    # Initialize the stream with the other embeddings, to add drift
    stream_untrained = WOSStream(
        transformer_model=transformer_model_untrained,
        transform=transform,
        test_split=False,
        device=device,
    )
    stream_untrained.prepare_for_use()

    # Load the model
    model = None
    stream_runner = None
    if lstm_model_idx is None and nb_model_idx is None:
        raise ValueError("No index provided for either the LSTM or the NB model.")
    if lstm_model_idx is not None:
        # Load the LSTM model
        model = LSTM(
            embedding_dim=utils.EMBEDDING_DIM, no_classes=stream_trained.n_classes
        ).to(device)
        model.load_state_dict(
            torch.load(LSTM_MODELS[lstm_model_idx], map_location=device), strict=False
        )
        model.eval()
        stream_runner = run_stream_lstm
    elif nb_model_idx is not None:
        # Load the Naive Bayes model
        model = load(NB_MODELS[nb_model_idx])
        stream_runner = run_stream_nb

    # Initialize the drift detector
    drift_detector = DDM()

    # Run streams
    print("Running trained stream...")
    trained_accuracies = stream_runner(
        stream_trained,
        model,
        drift_detector,
        batch_size=batch_size,
        print_every=print_every,
        warm_start=sys.maxsize,
        device=device,
    )
    print("Running untrained stream...")
    untrained_accuracies = stream_runner(
        stream_untrained,
        model,
        drift_detector,
        batch_size=batch_size,
        print_every=print_every,
        warm_start=sys.maxsize,
        device=device,
    )
    print("Fine-tuning...")
    stream_untrained.restart()
    fine_tuned_accuracies = fine_tune(
        model, stream_untrained, no_batches=no_batches, device=device,
    )

    # Save the results
    to_save = {
        "trained_accuracies": trained_accuracies,
        "untrained_accuracies": untrained_accuracies,
        "fine_tuned_accuracies": fine_tuned_accuracies,
    }

    with open(os.path.join(PATH_RESULTS, save_name + ".pkl"), "wb") as f:
        pickle.dump(to_save, f)

    return to_save


if __name__ == "__main__":
    fine_tuning_different_embeddings(
        "fine_tuning_lstm_wos_1_BERT_DISTILBERT_100_batches",
        lstm_model_idx=0,
        nb_model_idx=None,
        batch_size=32,
        no_batches=100,
        transformer_model_trained=TransformerModel.BERT,
        transformer_model_untrained=TransformerModel.DISTILBERT,
        print_every=1,
        transform=False,
        device="cpu",
    )
