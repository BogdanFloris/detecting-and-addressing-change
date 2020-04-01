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
ABRUPT_DRIFT_RESULT = [
    os.path.join(PATH_RESULTS, "diff_embed_lstm_wos_1_BERT_SCIBERT.pkl"),
    os.path.join(PATH_RESULTS, "diff_embed_lstm_wos_1_BERT_DISTILBERT.pkl"),
    os.path.join(PATH_RESULTS, "diff_embed_lstm_wos_1_BERT_SCIBERT_unsupervised.pkl"),
    os.path.join(
        PATH_RESULTS, "diff_embed_lstm_wos_1_BERT_DISTILBERT_unsupervised.pkl"
    ),
    os.path.join(PATH_RESULTS, "diff_embed_nb_wos_1_BERT_SCIBERT.pkl"),
    os.path.join(PATH_RESULTS, "diff_embed_nb_wos_1_BERT_DISTILBERT.pkl"),
    os.path.join(PATH_RESULTS, "diff_embed_nb_wos_1_BERT_SCIBERT_unsupervised.pkl"),
    os.path.join(PATH_RESULTS, "diff_embed_nb_wos_1_BERT_DISTILBERT_unsupervised.pkl"),
]
GRADUAL_DRIFT_RESULT = [
    os.path.join(PATH_RESULTS, "gradual_noise_random_std_1_lstm_wos_1_BERT.pkl"),
    os.path.join(PATH_RESULTS, "gradual_noise_random_std_2_lstm_wos_1_BERT.pkl"),
    os.path.join(PATH_RESULTS, "gradual_noise_random_std_3_lstm_wos_1_BERT.pkl"),
    os.path.join(PATH_RESULTS, "gradual_noise_random_std_1_nb_wos_1_BERT.pkl"),
    os.path.join(PATH_RESULTS, "gradual_noise_random_std_2_nb_wos_1_BERT.pkl"),
    os.path.join(PATH_RESULTS, "gradual_noise_random_std_3_nb_wos_1_BERT.pkl"),
]
FINE_TUNING_RESULT = [
    os.path.join(PATH_RESULTS, "fine_tuning_lstm_wos_1_BERT_DISTILBERT_50_batches.pkl"),
    os.path.join(
        PATH_RESULTS, "fine_tuning_lstm_wos_1_BERT_DISTILBERT_100_batches.pkl"
    ),
]
PROCRUSTES_RESULT = [
    os.path.join(PATH_RESULTS, "procrustes_lstm_wos_1_BERT_SCIBERT_5000_words_max.pkl"),
    os.path.join(
        PATH_RESULTS, "procrustes_lstm_wos_1_BERT_SCIBERT_5000_words_first.pkl"
    ),
    os.path.join(
        PATH_RESULTS, "procrustes_lstm_wos_1_BERT_SCIBERT_5000_words_average.pkl"
    ),
]


def visualize_mapping(mapping_idx, title, filename_path):
    with open(PROCRUSTES_RESULT[mapping_idx], "rb") as f:
        results: dict = pickle.load(f)

    accuracies = (
        results["trained_accuracies"]
        + results["untrained_accuracies"]
        + results["mapping_accuracies"]
    )
    trained_accuracies = [acc for acc, _ in results["trained_accuracies"]]
    untrained_accuracies = [acc for acc, _ in results["untrained_accuracies"]]
    mapping_accuracies = results["mapping_accuracies"]
    mapping_start_index = len(trained_accuracies) + len(untrained_accuracies)

    time_idx, filtered_acc, detections = [], [], []
    has_w, has_d = False, False
    for i in range(len(accuracies)):
        if i >= mapping_start_index:
            acc, det = accuracies[i], "N"
        else:
            acc, det = accuracies[i]
        if det != "N":
            time_idx.append(i)
            filtered_acc.append(acc)
            if det == "W":
                has_w = True
                detections.append("Warning")
            else:
                has_d = True
                detections.append("Drift")

    df_line = pd.DataFrame(
        dict(
            time=np.arange(len(accuracies)),
            accuracy=trained_accuracies + untrained_accuracies + mapping_accuracies,
            stream=["trained"] * len(trained_accuracies)
            + ["untrained"] * len(untrained_accuracies)
            + ["mapping"] * len(mapping_accuracies),
        )
    )
    df_scatter = pd.DataFrame(
        dict(time=time_idx, accuracy=filtered_acc, detection=detections,)
    )
    sns.set(style="darkgrid")

    fig, ax = plt.subplots()

    sns.lineplot(
        x="time",
        y="accuracy",
        hue="stream",
        data=df_line,
        alpha=0.5,
        palette=sns.xkcd_palette(
            colors=["denim blue", "medium green", "medium purple"]
        ),
        linewidth=0.8,
        ax=ax,
    )

    scatter_palette = []
    sizes = []
    if has_w:
        scatter_palette.append("amber")
        sizes.append(50.0)
    if has_d:
        scatter_palette.append("pale red")
        sizes.append(100.0)
    sns.scatterplot(
        x="time",
        y="accuracy",
        hue="detection",
        size="detection",
        sizes=sizes,
        marker="X",
        palette=sns.xkcd_palette(colors=scatter_palette),
        data=df_scatter,
        ax=ax,
    )

    plt.title(title)
    plt.ylim((-0.05, 1.05))
    plt.tight_layout()
    plt.savefig(filename_path)
    plt.show()


def visualize_fine_tuning(fine_tuning_idx, title, filename_path):
    with open(FINE_TUNING_RESULT[fine_tuning_idx], "rb") as f:
        results: dict = pickle.load(f)

    accuracies = (
        results["trained_accuracies"]
        + results["untrained_accuracies"]
        + results["fine_tuned_accuracies"]
    )
    trained_accuracies = [acc for acc, _ in results["trained_accuracies"]]
    untrained_accuracies = [acc for acc, _ in results["untrained_accuracies"]]
    fine_tuned_accuracies = results["fine_tuned_accuracies"]
    fine_tuned_start_index = len(trained_accuracies) + len(untrained_accuracies)

    time_idx, filtered_acc, detections = [], [], []
    has_w, has_d = False, False
    for i in range(len(accuracies)):
        if i >= fine_tuned_start_index:
            acc, det = accuracies[i], "N"
        else:
            acc, det = accuracies[i]
        if det != "N":
            time_idx.append(i)
            filtered_acc.append(acc)
            if det == "W":
                has_w = True
                detections.append("Warning")
            else:
                has_d = True
                detections.append("Drift")

    df_line = pd.DataFrame(
        dict(
            time=np.arange(len(accuracies)),
            accuracy=trained_accuracies + untrained_accuracies + fine_tuned_accuracies,
            stream=["trained"] * len(trained_accuracies)
            + ["untrained"] * len(untrained_accuracies)
            + ["fine tuned"] * len(fine_tuned_accuracies),
        )
    )
    df_scatter = pd.DataFrame(
        dict(time=time_idx, accuracy=filtered_acc, detection=detections,)
    )
    sns.set(style="darkgrid")

    fig, ax = plt.subplots()

    sns.lineplot(
        x="time",
        y="accuracy",
        hue="stream",
        data=df_line,
        alpha=0.5,
        palette=sns.xkcd_palette(
            colors=["denim blue", "medium green", "medium purple"]
        ),
        linewidth=0.8,
        ax=ax,
    )

    scatter_palette = []
    sizes = []
    if has_w:
        scatter_palette.append("amber")
        sizes.append(50.0)
    if has_d:
        scatter_palette.append("pale red")
        sizes.append(100.0)
    sns.scatterplot(
        x="time",
        y="accuracy",
        hue="detection",
        size="detection",
        sizes=sizes,
        marker="X",
        palette=sns.xkcd_palette(colors=scatter_palette),
        data=df_scatter,
        ax=ax,
    )

    plt.title(title)
    plt.ylim((-0.05, 1.05))
    plt.tight_layout()
    plt.savefig(filename_path)
    plt.show()


def visualize_abrupt_drift(drift_idx, title, filename_path):
    with open(ABRUPT_DRIFT_RESULT[drift_idx], "rb") as f:
        results: dict = pickle.load(f)

    accuracies = results["trained_accuracies"] + results["untrained_accuracies"]
    trained_accuracies = [acc for acc, _ in results["trained_accuracies"]]
    untrained_accuracies = [acc for acc, _ in results["untrained_accuracies"]]
    time_idx, filtered_acc, detections = [], [], []
    has_w, has_d = False, False
    for i, (acc, det) in enumerate(accuracies):
        if det != "N":
            time_idx.append(i)
            filtered_acc.append(acc)
            if det == "W":
                has_w = True
                detections.append("Warning")
            else:
                has_d = True
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
        dict(time=time_idx, accuracy=filtered_acc, detection=detections,)
    )
    sns.set(style="darkgrid")

    fig, ax = plt.subplots()

    sns.lineplot(
        x="time",
        y="accuracy",
        hue="stream",
        data=df_line,
        alpha=0.5,
        palette=sns.xkcd_palette(colors=["denim blue", "medium green"]),
        linewidth=0.8,
        ax=ax,
    )
    scatter_palette = []
    sizes = []
    if has_w:
        scatter_palette.append("amber")
        sizes.append(50.0)
    if has_d:
        scatter_palette.append("pale red")
        sizes.append(100.0)
    sns.scatterplot(
        x="time",
        y="accuracy",
        hue="detection",
        size="detection",
        sizes=sizes,
        marker="X",
        palette=sns.xkcd_palette(colors=scatter_palette),
        data=df_scatter,
        ax=ax,
    )
    plt.title(title)
    plt.ylim((-0.05, 1.05))
    plt.tight_layout()
    plt.savefig(filename_path)
    plt.show()


def visualize_gradual_drift(drift_idx, title, filename_path):
    with open(GRADUAL_DRIFT_RESULT[drift_idx], "rb") as f:
        results: dict = pickle.load(f)

    accuracies = [acc for acc, _ in results["accuracies"]]
    time_idx, filtered_acc, detections = [], [], []
    for i, (acc, det) in enumerate(results["accuracies"]):
        if det != "N":
            time_idx.append(i)
            filtered_acc.append(acc)
            if det == "W":
                detections.append("Warning")
            else:
                detections.append("Drift")

    df_line = pd.DataFrame(dict(time=np.arange(len(accuracies)), accuracy=accuracies,))
    df_scatter = pd.DataFrame(
        dict(time=time_idx, accuracy=filtered_acc, detection=detections,)
    )
    sns.set(style="darkgrid")

    fig, ax = plt.subplots()

    sns.lineplot(
        x="time", y="accuracy", data=df_line, alpha=0.6, linewidth=0.8, ax=ax,
    )
    sns.scatterplot(
        x="time",
        y="accuracy",
        hue="detection",
        size="detection",
        sizes=[50.0, 100.0],
        marker="X",
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
    # visualize_abrupt_drift(
    #     7,
    #     "Concept drift over time unsupervised (BERT-DISTILBERT streams on LSTM model)",
    #     os.path.join(
    #         PATH_FIGURES, "diff_embed_nb_wos_1_BERT_DISTILBERT_unsupervised.png"
    #     ),
    # )
    # visualize_gradual_drift(
    #     5,
    #     "Gradual drift over time (random noise max std 3.0, BERT, Naive Bayes)",
    #     os.path.join(PATH_FIGURES, "gradual_noise_random_std_3_nb_wos_1_BERT.png"),
    # )
    # visualize_fine_tuning(
    #     1,
    #     "Fine tuning (100 batches) accuracy after small abrupt drift (BERT-DISTILBERT-LSTM)",
    #     os.path.join(
    #         PATH_FIGURES, "fine_tuning_lstm_wos_1_BERT_DISTILBERT_100_batches.png"
    #     ),
    # )
    visualize_mapping(
        2,
        "Adaptation using Procrustes mapping after abrupt drift (BERT-SCIBERT-LSTM)",
        os.path.join(
            PATH_FIGURES, "procrustes_lstm_wos_1_BERT_SCIBERT_5000_words_average.png"
        ),
    )
