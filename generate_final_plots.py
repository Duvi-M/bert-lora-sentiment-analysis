import csv
import os
import re
from pathlib import Path

MATPLOTLIB_CONFIG_DIR = Path("outputs/final_plots/.matplotlib")
MATPLOTLIB_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MATPLOTLIB_CONFIG_DIR))

import matplotlib.pyplot as plt


EXPERIMENTS = [
    {
        "experiment_name": "fair_budget_96",
        "budget": 96,
        "path": Path("outputs/fair_budget_96"),
    },
    {
        "experiment_name": "fair_budget_144",
        "budget": 144,
        "path": Path("outputs/fair_budget_144"),
    },
    {
        "experiment_name": "fair_budget_192",
        "budget": 192,
        "path": Path("outputs/fair_budget_192"),
    },
]

FINAL_PLOTS_DIR = Path("outputs/final_plots")
RANK_VALUES = [2, 4, 6, 8]
ATTENTION_COLUMNS = ["query", "value"]
NUM_TRANSFORMER_LAYERS = 12


def shorten_layer_name(layer_name: str) -> str:
    """Convert long BERT attention module names into compact thesis labels."""
    match = re.search(r"layer\.(\d+).*\.([^.]+)$", layer_name)
    if not match:
        return layer_name

    layer_idx, module_name = match.groups()
    module_short = {
        "query": "Q",
        "value": "V",
    }.get(module_name, module_name[:1].upper())

    return f"L{layer_idx}-{module_short}"


def read_single_row_csv(path: Path) -> dict:
    with path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    return rows[0] if rows else {}


def read_key_value_csv(path: Path, key_field: str, value_field: str, value_type=float):
    values = {}
    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            values[row[key_field]] = value_type(row[value_field])
    return values


def read_table_csv(path: Path):
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def load_experiment_result(experiment: dict):
    experiment_name = experiment["experiment_name"]
    budget = experiment["budget"]
    output_dir = experiment["path"]

    if not output_dir.exists():
        print(
            f"Warning: missing experiment folder for {experiment_name}, "
            f"skipping: {output_dir}"
        )
        return None

    metrics_path = output_dir / "metrics.csv"
    rank_pattern_path = output_dir / "rank_pattern.csv"
    layer_scores_path = output_dir / "layer_scores.csv"
    allocation_history_path = output_dir / "allocation_history.csv"

    missing_files = [
        path
        for path in [metrics_path, rank_pattern_path, layer_scores_path]
        if not path.exists()
    ]
    if missing_files:
        for path in missing_files:
            print(f"Warning: missing file, skipping budget {budget}: {path}")
        return None

    metrics = read_single_row_csv(metrics_path)
    rank_pattern = read_key_value_csv(
        rank_pattern_path,
        key_field="layer",
        value_field="rank",
        value_type=int,
    )
    layer_scores = read_key_value_csv(
        layer_scores_path,
        key_field="layer",
        value_field="score",
        value_type=float,
    )
    allocation_history = []
    if allocation_history_path.exists():
        allocation_history = read_table_csv(allocation_history_path)
    else:
        print(
            f"Warning: optional allocation history not found for "
            f"{experiment_name}: {allocation_history_path}"
        )

    return {
        "experiment_name": experiment_name,
        "budget": budget,
        "output_dir": str(output_dir),
        "metrics": metrics,
        "rank_pattern": rank_pattern,
        "layer_scores": layer_scores,
        "allocation_history": allocation_history,
    }


def to_float(value):
    if value in (None, ""):
        return None
    return float(value)


def to_int(value):
    if value in (None, ""):
        return None
    return int(float(value))


def percent_reduction(baseline_value, adaptive_value):
    baseline = to_float(baseline_value)
    adaptive = to_float(adaptive_value)

    if baseline in (None, 0) or adaptive is None:
        return None

    return 100.0 * (baseline - adaptive) / baseline


def accuracy_drop(baseline_value, adaptive_value):
    baseline = to_float(baseline_value)
    adaptive = to_float(adaptive_value)

    if baseline is None or adaptive is None:
        return None

    return baseline - adaptive


def format_optional_float(value):
    if value is None:
        return ""
    return f"{value:.6f}"


def extract_layer_and_module(layer_name: str):
    match = re.search(r"layer\.(\d+).*\.([^.]+)$", layer_name)
    if not match:
        return None, None

    layer_idx = int(match.group(1))
    module_name = match.group(2)

    if module_name not in ATTENTION_COLUMNS:
        return None, None

    return layer_idx, module_name


def build_layer_matrix(values: dict, default_value=0.0):
    matrix = [
        [default_value for _ in ATTENTION_COLUMNS]
        for _ in range(NUM_TRANSFORMER_LAYERS)
    ]

    for layer_name, value in values.items():
        layer_idx, module_name = extract_layer_and_module(layer_name)
        if layer_idx is None or layer_idx >= NUM_TRANSFORMER_LAYERS:
            continue

        column_idx = ATTENTION_COLUMNS.index(module_name)
        matrix[layer_idx][column_idx] = value

    return matrix


def save_final_summary(results: list):
    FINAL_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = FINAL_PLOTS_DIR / "final_summary.csv"

    fieldnames = [
        "experiment_name",
        "budget",
        "baseline_accuracy",
        "baseline_loss",
        "baseline_trainable_params",
        "adaptive_accuracy",
        "adaptive_loss",
        "adaptive_trainable_params",
        "parameter_reduction_percent",
        "accuracy_drop",
    ]

    with summary_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            metrics = result["metrics"]
            reduction = percent_reduction(
                metrics.get("baseline_trainable_params"),
                metrics.get("adaptive_trainable_params"),
            )
            drop = accuracy_drop(
                metrics.get("baseline_accuracy"),
                metrics.get("adaptive_accuracy"),
            )
            writer.writerow({
                "experiment_name": result["experiment_name"],
                "budget": result["budget"],
                "baseline_accuracy": metrics.get("baseline_accuracy"),
                "baseline_loss": metrics.get("baseline_loss"),
                "baseline_trainable_params": metrics.get(
                    "baseline_trainable_params"
                ),
                "adaptive_accuracy": metrics.get("adaptive_accuracy"),
                "adaptive_loss": metrics.get("adaptive_loss"),
                "adaptive_trainable_params": metrics.get(
                    "adaptive_trainable_params"
                ),
                "parameter_reduction_percent": format_optional_float(reduction),
                "accuracy_drop": format_optional_float(drop),
            })

    print(f"Saved summary: {summary_path}")


def count_ranks(rank_pattern: dict):
    counts = {rank: 0 for rank in RANK_VALUES}
    for rank in rank_pattern.values():
        if rank in counts:
            counts[rank] += 1
    return counts


def plot_accuracy_vs_budget(results: list):
    """Plot Adaptive LoRA accuracy across budgets, with baseline points."""
    budgets = [result["budget"] for result in results]
    adaptive_accuracy = [
        to_float(result["metrics"].get("adaptive_accuracy"))
        for result in results
    ]
    baseline_accuracy = [
        to_float(result["metrics"].get("baseline_accuracy"))
        for result in results
    ]

    plt.figure(figsize=(8, 5))
    plt.plot(
        budgets,
        adaptive_accuracy,
        marker="o",
        linewidth=2,
        label="Adaptive LoRA",
    )
    plt.scatter(
        budgets,
        baseline_accuracy,
        marker="s",
        label="Fixed LoRA baseline",
    )
    plt.xlabel("Rank budget")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs rank budget")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FINAL_PLOTS_DIR / "accuracy_vs_budget.png", dpi=300)
    plt.close()


def plot_accuracy_vs_trainable_params(results: list):
    """Compare accuracy against trainable parameters for both methods."""
    baseline_params = [
        to_int(result["metrics"].get("baseline_trainable_params"))
        for result in results
    ]
    baseline_accuracy = [
        to_float(result["metrics"].get("baseline_accuracy"))
        for result in results
    ]
    adaptive_params = [
        to_int(result["metrics"].get("adaptive_trainable_params"))
        for result in results
    ]
    adaptive_accuracy = [
        to_float(result["metrics"].get("adaptive_accuracy"))
        for result in results
    ]

    plt.figure(figsize=(8, 5))
    plt.scatter(
        baseline_params,
        baseline_accuracy,
        marker="s",
        label="Fixed LoRA baseline",
    )
    plt.plot(
        adaptive_params,
        adaptive_accuracy,
        marker="o",
        linewidth=2,
        label="Adaptive LoRA",
    )
    plt.xlabel("Trainable parameters")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs trainable parameters")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FINAL_PLOTS_DIR / "accuracy_vs_trainable_params.png", dpi=300)
    plt.close()


def plot_rank_distribution_by_budget(results: list):
    """Show how many LoRA modules receive each rank for every budget."""
    budgets = [result["budget"] for result in results]
    x_positions = list(range(len(budgets)))
    bar_width = 0.18

    plt.figure(figsize=(9, 5))
    for offset_idx, rank in enumerate(RANK_VALUES):
        counts = [
            count_ranks(result["rank_pattern"]).get(rank, 0)
            for result in results
        ]
        positions = [
            x + (offset_idx - 1.5) * bar_width
            for x in x_positions
        ]
        plt.bar(positions, counts, width=bar_width, label=f"Rank {rank}")

    plt.xticks(x_positions, budgets)
    plt.xlabel("Rank budget")
    plt.ylabel("Number of LoRA modules")
    plt.title("Adaptive rank distribution by budget")
    plt.grid(axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FINAL_PLOTS_DIR / "rank_distribution_by_budget.png", dpi=300)
    plt.close()


def plot_heatmap(matrix, title: str, output_name: str, colorbar_label: str):
    plt.figure(figsize=(5, 7))
    image = plt.imshow(matrix, aspect="auto", cmap="viridis")
    plt.colorbar(image, label=colorbar_label)
    plt.xticks(range(len(ATTENTION_COLUMNS)), ["Query", "Value"])
    plt.yticks(
        range(NUM_TRANSFORMER_LAYERS),
        [f"Layer {idx}" for idx in range(NUM_TRANSFORMER_LAYERS)],
    )
    plt.xlabel("Attention module")
    plt.ylabel("Transformer layer")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(FINAL_PLOTS_DIR / output_name, dpi=300)
    plt.close()


def plot_rank_heatmaps(results_by_budget: dict):
    """Create one rank heatmap per budget for query/value modules."""
    for budget in [96, 144, 192]:
        result = results_by_budget.get(budget)
        if result is None:
            print(f"Warning: cannot create rank heatmap for budget {budget}")
            continue

        matrix = build_layer_matrix(result["rank_pattern"], default_value=0)
        plot_heatmap(
            matrix=matrix,
            title=f"Adaptive rank heatmap, budget {budget}",
            output_name=f"rank_heatmap_budget_{budget}.png",
            colorbar_label="Assigned rank",
        )


def plot_gradient_score_heatmap_budget_96(results_by_budget: dict):
    """Create the gradient score heatmap for the budget-96 scoring run."""
    result = results_by_budget.get(96)
    if result is None:
        print("Warning: cannot create gradient score heatmap for budget 96")
        return

    matrix = build_layer_matrix(result["layer_scores"], default_value=0.0)
    plot_heatmap(
        matrix=matrix,
        title="Gradient importance score heatmap, budget 96",
        output_name="gradient_score_heatmap_budget_96.png",
        colorbar_label="Gradient importance score",
    )


def main():
    FINAL_PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    results = []
    for experiment in EXPERIMENTS:
        result = load_experiment_result(experiment)
        if result is not None:
            results.append(result)

    save_final_summary(results)

    if not results:
        expected_folders = ", ".join(
            str(experiment["path"])
            for experiment in EXPERIMENTS
        )
        print("Warning: no valid fair-comparison results found.")
        print(f"Expected folders: {expected_folders}")
        print("No plots were generated.")
        return

    results = sorted(results, key=lambda item: item["budget"])
    results_by_budget = {
        result["budget"]: result
        for result in results
    }

    plot_accuracy_vs_budget(results)
    plot_accuracy_vs_trainable_params(results)
    plot_rank_distribution_by_budget(results)
    plot_rank_heatmaps(results_by_budget)
    plot_gradient_score_heatmap_budget_96(results_by_budget)

    print(f"Saved final plots to: {FINAL_PLOTS_DIR}")


if __name__ == "__main__":
    main()
