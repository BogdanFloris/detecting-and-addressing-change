"""
Experiments with drift detection
"""
import os
import torch
import utils
from pathlib import Path
from skmultiflow.drift_detection import DDM, EDDM
from streams.stream_data import WOSStream
from models.wos_classifier import LSTM
from constants.transformers import TransformerModel
from utils.metrics import accuracy


PATH = os.path.join(Path(__file__).parents[1], "assets/models")
MODELS = [
    os.path.join(PATH, "lstm-wos-BERT-ver-1-batch/model.pt"),
]


def drift_detection_gradual_noise(
    model_idx=0, batch_size=1, max_std=0.1, warm_start=30, print_every=10, device="cpu"
):
    # Initialize the stream
    stream = WOSStream(transformer_model=TransformerModel.BERT)
    stream.prepare_for_use()

    # Load the LSTM model
    model = LSTM(embedding_dim=utils.EMBEDDING_DIM, no_classes=stream.n_classes).to(
        device
    )
    model.load_state_dict(
        torch.load(MODELS[model_idx], map_location=device), strict=False
    )
    model.eval()

    # Initialize the drift detector
    eddm = EDDM()

    # Run stream
    running_acc = 0.0
    n_iterations = stream.n_samples // batch_size + 1
    standard_devs = torch.arange(
        start=0, end=max_std, step=max_std / (n_iterations - warm_start)
    )
    for i in range(n_iterations):
        # Get the batch from the stream
        if stream.n_remaining_samples() < batch_size:
            x_, y = stream.next_sample(stream.n_remaining_samples())
        else:
            x_, y = stream.next_sample(batch_size)

        x, seq_lens = x_
        if i >= warm_start:
            # Construct and add noise to x
            std = torch.zeros_like(x) + standard_devs[i - warm_start]
            noise = torch.normal(0, std)
            x = x + noise
        # Move the batch to device
        x = x.to(device)
        y = torch.from_numpy(y).to(device)
        seq_lens = torch.tensor(seq_lens).to(device)

        # Get predictions and accuracy
        predictions, _ = model((x, seq_lens))
        acc = accuracy(labels=y, predictions=predictions).item()

        # Print if necessary
        running_acc += acc
        if i % print_every == print_every - 1:
            print("Accuracy: {}".format(running_acc / print_every))
            running_acc = 0.0

        # Add to drift detector
        eddm.add_element(1 - acc)
        if eddm.detected_warning_zone():
            print("Warning zone")
        if eddm.detected_change():
            print("Drift detected")


def drift_detection_different_embeddings(
    model_idx=0, batch_size=1, print_every=10, device="cpu"
):
    # Initialize the stream that the model was trained on
    stream_trained = WOSStream(transformer_model=TransformerModel.BERT)
    stream_trained.prepare_for_use()
    # Initialize the stream with other embeddings, to add drift
    stream_untrained = WOSStream(transformer_model=TransformerModel.SCIBERT)
    stream_trained.prepare_for_use()

    # Load the LSTM model
    model = LSTM(
        embedding_dim=utils.EMBEDDING_DIM, no_classes=stream_trained.n_classes
    ).to(device)
    model.load_state_dict(
        torch.load(MODELS[model_idx], map_location=device), strict=False
    )
    model.eval()

    # Initialize drift detector
    ddm = DDM()

    # Run stream
    print("Running trained stream...")
    run_stream(
        stream_trained,
        model,
        ddm,
        batch_size=batch_size,
        print_every=print_every,
        device=device,
    )
    print("Running untrained stream...")
    run_stream(
        stream_untrained,
        model,
        ddm,
        batch_size=batch_size,
        print_every=print_every,
        device=device,
    )


def run_stream(
    stream, model, drift_detector, batch_size=1, print_every=10, device="cpu"
):
    i = 0
    running_acc = 0.0
    while stream.has_more_samples():
        # Get the batch from the stream
        if stream.n_remaining_samples() < batch_size:
            x_, y = stream.next_sample(stream.n_remaining_samples())
        else:
            x_, y = stream.next_sample(batch_size)

        # Move the batch to device
        x, seq_lens = x_
        x = x.to(device)
        y = torch.from_numpy(y).to(device)
        seq_lens = torch.tensor(seq_lens).to(device)

        # Get predictions and accuracy
        predictions, _ = model((x, seq_lens))
        acc = accuracy(labels=y, predictions=predictions).item()

        # Print if necessary
        running_acc += acc
        if i % print_every == print_every - 1:
            print("Accuracy: {}".format(running_acc / print_every))
            running_acc = 0.0

        # Add to drift detector
        drift_detector.add_element(1 - acc)
        if drift_detector.detected_warning_zone():
            print("Warning zone")
        if drift_detector.detected_change():
            print("Drift detected")

        i += 1


if __name__ == "__main__":
    drift_detection_different_embeddings(batch_size=32)
