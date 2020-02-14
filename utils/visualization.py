"""
Visualization methods
"""
import os
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

PATH_RESULTS = os.path.join(Path(__file__).parents[1], "assets/results")
PATH_FIGURES = os.path.join(Path(__file__).parents[1], "assets/figures")
DRIFT_RESULT = [
    os.path.join(PATH_RESULTS, "diff_embed_lstm_wos_1_BERT_SCIBERT.pkl"),
    os.path.join(PATH_RESULTS, "diff_embed_lstm_wos_1_BERT_DISTILBERT.pkl"),
]


def visualize_drift(drift_idx, title, filename_path):
    with open(DRIFT_RESULT[drift_idx], "rb") as f:
        results: dict = pickle.load(f)

    type_colors = ["amber", "pale red"]
    accuracies = results["trained_accuracies"] + results["untrained_accuracies"]
    trained_accuracies = [acc for acc, _ in results["trained_accuracies"]]
    untrained_accuracies = [acc for acc, _ in results["untrained_accuracies"]]
    time_idx, filtered_acc, detections = [], [], []
    for i, (acc, det) in enumerate(accuracies):
        if det != "N":
            time_idx.append(i)
            filtered_acc.append(acc)
            if det == "W":
                detections.append("Warning")
            else:
                detections.append("Drift")

    df_line = pd.DataFrame(
        dict(
            time=np.arange(len(trained_accuracies) + len(untrained_accuracies)),
            accuracy=trained_accuracies + untrained_accuracies,
            stream=["trained"] * len(trained_accuracies)
            + ["untrained"] * len(untrained_accuracies),
        )
    )
    df_scatter = pd.DataFrame(
        dict(
            time=time_idx,
            accuracy=filtered_acc,
            detection=detections,
        )
    )
    sns.set(style="darkgrid")

    fig, ax = plt.subplots()

    sns.lineplot(
        x="time",
        y="accuracy",
        hue="stream",
        data=df_line,
        alpha=0.4,
        palette=sns.xkcd_palette(colors=["denim blue", "medium green"]),
        linewidth=0.8,
        ax=ax,
    )
    sns.scatterplot(
        x="time",
        y="accuracy",
        hue="detection",
        size="detection",
        sizes=[50.0, 100.0],
        marker='X',
        palette=sns.xkcd_palette(colors=["amber", "pale red"]),
        data=df_scatter,
        ax=ax,
    )
    plt.title(title)
    plt.ylim((-0.05, 1.05))
    plt.tight_layout()
    plt.savefig(filename_path)
    plt.show()


if __name__ == "__main__":
    visualize_drift(
        0,
        "Concept drift over time (BERT-SCIBERT streams on LSTM model)",
        os.path.join(PATH_FIGURES, "diff_embed_lstm_wos_1_BERT_SCIBERT.png"),
    )
