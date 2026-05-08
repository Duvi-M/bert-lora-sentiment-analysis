import os
import pandas as pd
import matplotlib.pyplot as plt


def save_layer_scores_plot(layer_scores: dict, output_dir: str):
    df = pd.DataFrame(
        list(layer_scores.items()),
        columns=["layer", "score"],
    )

    csv_path = os.path.join(output_dir, "layer_scores.csv")
    df.to_csv(csv_path, index=False)

    plt.figure(figsize=(12, 5))
    plt.bar(df["layer"], df["score"])
    plt.xticks(rotation=90)
    plt.xlabel("Layer")
    plt.ylabel("Gradient score")
    plt.title("Layer importance scores")
    plt.tight_layout()

    fig_path = os.path.join(output_dir, "layer_scores.png")
    plt.savefig(fig_path)
    plt.close()


def save_rank_pattern_plot(rank_pattern: dict, output_dir: str):
    df = pd.DataFrame(
        list(rank_pattern.items()),
        columns=["layer", "rank"],
    )

    csv_path = os.path.join(output_dir, "rank_pattern.csv")
    df.to_csv(csv_path, index=False)

    plt.figure(figsize=(12, 5))
    plt.bar(df["layer"], df["rank"])
    plt.xticks(rotation=90)
    plt.xlabel("Layer")
    plt.ylabel("Assigned rank")
    plt.title("Adaptive rank distribution")
    plt.tight_layout()

    fig_path = os.path.join(output_dir, "rank_pattern.png")
    plt.savefig(fig_path)
    plt.close()


def save_allocation_history_plot(history: list, output_dir: str):
    df = pd.DataFrame(history)

    csv_path = os.path.join(output_dir, "allocation_history.csv")
    df.to_csv(csv_path, index=False)

    if df.empty:
        return

    plt.figure(figsize=(10, 5))
    plt.plot(df["iteration"], df["used_budget"], marker="o")
    plt.xlabel("Iteration")
    plt.ylabel("Used rank budget")
    plt.title("Budget allocation over iterations")
    plt.grid(True)
    plt.tight_layout()

    fig_path = os.path.join(output_dir, "budget_history.png")
    plt.savefig(fig_path)
    plt.close()


def save_metrics(metrics: dict, output_dir: str):
    df = pd.DataFrame([metrics])
    df.to_csv(os.path.join(output_dir, "metrics.csv"), index=False)