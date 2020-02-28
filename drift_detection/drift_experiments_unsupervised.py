"""
Experiments with drift detection (unsupervised).
"""
import os
import torch
import utils
import pickle
from joblib import load
from skmultiflow.drift_detection import DDM
from streams.stream_data import WOSStream
from streams.run_stream_unsupervised import run_lstm_streams, run_nb_streams
from models.wos_classifier import LSTM
from constants.transformers import TransformerModel
from drift_detection.drift_experiments_supervised import (
    LSTM_MODELS,
    NB_MODELS,
    PATH_RESULTS,
)


def drift_detection_different_embeddings(
    save_name,
    lstm_model_idx=None,
    nb_model_idx=None,
    transformer_model_trained=None,
    transformer_model_untrained=None,
    batch_size=1,
    transform=True,
    print_every=1,
    device="cpu",
):
    """
    Performs an experiment with two different streams on the same model.
    The first stream is the one with embeddings on which the model was trained on.
    The second stream is one with embeddings that are different from the ones
    on which the model was trained on.
    The goal of the experiment is to find if the new embeddings can be substituted
    for the old ones, which case no drift should occur, or otherwise they cannot be
    used and drift will be detected.
    This experiment regards the first stream as a baseline which outputs the ground truth
    labels and then the second stream is compared against it.

    Args:
        save_name (str): name of the file where the function saves the result
        lstm_model_idx (int): the index of the LSTM model (from the available ones)
        nb_model_idx (int): the index of the Naive Bayes model (from the available ones)
        transformer_model_trained (TransformerModel): the embeddings on which the model was trained
        transformer_model_untrained (TransformerModel): the embeddings against which the model is compared
        batch_size (int): the batch size for the stream
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
    # Initialize the stream with other embeddings, to add drift
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
        stream_runner = run_lstm_streams
    elif nb_model_idx is not None:
        # Load the Naive Bayes model
        model = load(NB_MODELS[nb_model_idx])
        stream_runner = run_nb_streams

    # Initialize drift detector
    drift_detector = DDM()

    # Run streams
    trained_accuracies, untrained_accuracies = stream_runner(
        stream_trained,
        stream_untrained,
        model,
        drift_detector,
        batch_size=batch_size,
        print_every=print_every,
        device=device,
    )

    # Save the results
    to_save = {
        "trained_accuracies": trained_accuracies,
        "untrained_accuracies": untrained_accuracies,
    }

    with open(os.path.join(PATH_RESULTS, save_name + ".pkl"), "wb") as f:
        pickle.dump(to_save, f)

    return to_save


if __name__ == "__main__":
    drift_detection_different_embeddings(
        "diff_embed_nb_wos_1_BERT_DISTILBERT_unsupervised",
        lstm_model_idx=None,
        nb_model_idx=0,
        batch_size=32,
        transformer_model_trained=TransformerModel.BERT,
        transformer_model_untrained=TransformerModel.DISTILBERT,
        print_every=1,
        transform=False,
        device="cpu",
    )
