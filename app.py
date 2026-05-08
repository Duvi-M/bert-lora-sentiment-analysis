from pathlib import Path

import pandas as pd
import streamlit as st
import yaml

from src.experiment import run_experiment
from src.utils import ensure_dir, load_config, set_seed


DEFAULT_CONFIG_PATH = Path("configs/default.yaml")


def load_default_config() -> dict:
    if DEFAULT_CONFIG_PATH.exists():
        return load_config(str(DEFAULT_CONFIG_PATH))

    return {
        "seed": 42,
        "output_dir": "outputs/run_streamlit",
        "dataset": {
            "name": "imdb",
            "train_size": 2000,
            "test_size": 1000,
            "max_length": 256,
        },
        "model": {
            "name": "bert-base-uncased",
            "num_labels": 2,
        },
        "training": {
            "epochs": 2,
            "batch_size": 8,
            "learning_rate": 5e-5,
            "logging_steps": 50,
        },
        "lora": {
            "alpha": 16,
            "dropout": 0.1,
            "target_modules": ["query", "value"],
        },
        "baseline_lora": {
            "r": 8,
        },
        "adaptive_lora": {
            "algorithm": "gradient_aware",
            "min_rank": 2,
            "max_rank": 8,
            "rank_step": 2,
            "total_budget": 96,
            "warmup_steps": 20,
        },
    }


def parse_target_modules(value: str) -> list[str]:
    return [module.strip() for module in value.split(",") if module.strip()]


def build_config_from_sidebar(default_config: dict) -> dict:
    st.sidebar.header("Experiment config")

    dataset_cfg = default_config["dataset"]
    model_cfg = default_config["model"]
    training_cfg = default_config["training"]
    lora_cfg = default_config["lora"]
    baseline_cfg = default_config["baseline_lora"]
    adaptive_cfg = default_config["adaptive_lora"]

    seed = st.sidebar.number_input("seed", min_value=0, value=int(default_config["seed"]))
    output_dir = st.sidebar.text_input("output_dir", value=default_config["output_dir"])

    st.sidebar.subheader("Dataset")
    dataset_name = st.sidebar.text_input("dataset.name", value=dataset_cfg["name"])
    train_size = st.sidebar.number_input(
        "dataset.train_size",
        min_value=1,
        value=int(dataset_cfg["train_size"]),
    )
    test_size = st.sidebar.number_input(
        "dataset.test_size",
        min_value=1,
        value=int(dataset_cfg["test_size"]),
    )
    max_length = st.sidebar.number_input(
        "dataset.max_length",
        min_value=1,
        value=int(dataset_cfg["max_length"]),
    )

    st.sidebar.subheader("Model")
    model_name = st.sidebar.text_input("model.name", value=model_cfg["name"])
    num_labels = st.sidebar.number_input(
        "model.num_labels",
        min_value=1,
        value=int(model_cfg["num_labels"]),
    )

    st.sidebar.subheader("Training")
    epochs = st.sidebar.number_input(
        "training.epochs",
        min_value=1,
        value=int(training_cfg["epochs"]),
    )
    batch_size = st.sidebar.number_input(
        "training.batch_size",
        min_value=1,
        value=int(training_cfg["batch_size"]),
    )
    learning_rate = st.sidebar.number_input(
        "training.learning_rate",
        min_value=0.0,
        value=float(training_cfg["learning_rate"]),
        format="%.8f",
    )
    logging_steps = st.sidebar.number_input(
        "training.logging_steps",
        min_value=1,
        value=int(training_cfg["logging_steps"]),
    )

    st.sidebar.subheader("LoRA")
    lora_alpha = st.sidebar.number_input(
        "lora.alpha",
        min_value=1,
        value=int(lora_cfg["alpha"]),
    )
    lora_dropout = st.sidebar.number_input(
        "lora.dropout",
        min_value=0.0,
        max_value=1.0,
        value=float(lora_cfg["dropout"]),
        format="%.3f",
    )
    target_modules = st.sidebar.text_input(
        "lora.target_modules",
        value=", ".join(lora_cfg["target_modules"]),
    )

    st.sidebar.subheader("Baseline LoRA")
    baseline_r = st.sidebar.number_input(
        "baseline_lora.r",
        min_value=1,
        value=int(baseline_cfg["r"]),
    )

    st.sidebar.subheader("Adaptive LoRA")
    adaptive_algorithm = st.sidebar.text_input(
        "adaptive_lora.algorithm",
        value=adaptive_cfg["algorithm"],
    )
    min_rank = st.sidebar.number_input(
        "adaptive_lora.min_rank",
        min_value=1,
        value=int(adaptive_cfg["min_rank"]),
    )
    max_rank = st.sidebar.number_input(
        "adaptive_lora.max_rank",
        min_value=1,
        value=int(adaptive_cfg["max_rank"]),
    )
    rank_step = st.sidebar.number_input(
        "adaptive_lora.rank_step",
        min_value=1,
        value=int(adaptive_cfg["rank_step"]),
    )
    total_budget = st.sidebar.number_input(
        "adaptive_lora.total_budget",
        min_value=1,
        value=int(adaptive_cfg["total_budget"]),
    )
    warmup_steps = st.sidebar.number_input(
        "adaptive_lora.warmup_steps",
        min_value=1,
        value=int(adaptive_cfg["warmup_steps"]),
    )

    return {
        "seed": int(seed),
        "output_dir": output_dir,
        "dataset": {
            "name": dataset_name,
            "train_size": int(train_size),
            "test_size": int(test_size),
            "max_length": int(max_length),
        },
        "model": {
            "name": model_name,
            "num_labels": int(num_labels),
        },
        "training": {
            "epochs": int(epochs),
            "batch_size": int(batch_size),
            "learning_rate": float(learning_rate),
            "logging_steps": int(logging_steps),
        },
        "lora": {
            "alpha": int(lora_alpha),
            "dropout": float(lora_dropout),
            "target_modules": parse_target_modules(target_modules),
        },
        "baseline_lora": {
            "r": int(baseline_r),
        },
        "adaptive_lora": {
            "algorithm": adaptive_algorithm,
            "min_rank": int(min_rank),
            "max_rank": int(max_rank),
            "rank_step": int(rank_step),
            "total_budget": int(total_budget),
            "warmup_steps": int(warmup_steps),
        },
    }


def show_metric_cards(title: str, metrics: dict) -> None:
    st.subheader(title)
    cols = st.columns(4)
    cols[0].metric("Accuracy", format_metric(metrics.get("accuracy")))
    cols[1].metric("Loss", format_metric(metrics.get("loss")))
    cols[2].metric("Train time", format_seconds(metrics.get("train_time")))
    cols[3].metric("Trainable params", format_int(metrics.get("trainable_params")))

    total_params = metrics.get("total_params")
    if total_params is not None:
        st.caption(f"Total parameters: {format_int(total_params)}")


def format_metric(value) -> str:
    if value is None:
        return "N/A"
    return f"{value:.4f}"


def format_seconds(value) -> str:
    if value is None:
        return "N/A"
    return f"{value:.1f}s"


def format_int(value) -> str:
    if value is None:
        return "N/A"
    return f"{int(value):,}"


def show_comparison(results: dict) -> None:
    baseline = results["baseline"]
    adaptive = results["adaptive"]

    st.subheader("Accuracy and loss comparison")
    comparison_df = pd.DataFrame(
        [
            {
                "method": "baseline_lora",
                "accuracy": baseline.get("accuracy"),
                "loss": baseline.get("loss"),
                "trainable_params": baseline.get("trainable_params"),
            },
            {
                "method": "adaptive_lora",
                "accuracy": adaptive.get("accuracy"),
                "loss": adaptive.get("loss"),
                "trainable_params": adaptive.get("trainable_params"),
            },
        ]
    )
    st.dataframe(comparison_df, use_container_width=True)
    st.bar_chart(comparison_df.set_index("method")[["accuracy", "loss"]])


def show_csv_if_available(output_dir: Path, filename: str, title: str) -> None:
    path = output_dir / filename
    if path.exists():
        st.subheader(title)
        st.dataframe(pd.read_csv(path), use_container_width=True)


def show_image_if_available(output_dir: Path, filename: str, caption: str) -> None:
    path = output_dir / filename
    if path.exists():
        st.subheader(caption)
        st.image(str(path), use_container_width=True)


def show_saved_outputs(output_dir: str) -> None:
    output_path = Path(output_dir)

    show_csv_if_available(output_path, "layer_scores.csv", "Layer scores")
    show_csv_if_available(output_path, "rank_pattern.csv", "Rank pattern")
    show_csv_if_available(output_path, "allocation_history.csv", "Allocation history")

    show_image_if_available(output_path, "layer_scores.png", "Layer scores plot")
    show_image_if_available(output_path, "rank_pattern.png", "Rank pattern plot")
    show_image_if_available(output_path, "budget_history.png", "Budget history plot")


def main() -> None:
    st.set_page_config(page_title="DULoRA", layout="wide")
    st.title("DULoRA: Dynamic Utility-based LoRA Rank Allocation")

    default_config = load_default_config()
    config = build_config_from_sidebar(default_config)

    with st.expander("Current config"):
        st.code(yaml.safe_dump(config, sort_keys=False), language="yaml")

    if st.button("Run experiment", type="primary"):
        ensure_dir(config["output_dir"])
        set_seed(config["seed"])

        with st.spinner("Running DULoRA experiment..."):
            results = run_experiment(config)

        st.success(f"Experiment finished. Outputs saved to {config['output_dir']}")
        show_metric_cards("Baseline LoRA metrics", results["baseline"])
        show_metric_cards("Adaptive LoRA metrics", results["adaptive"])
        show_comparison(results)
        show_saved_outputs(config["output_dir"])
    else:
        st.info("Configure the experiment in the sidebar, then click Run experiment.")


if __name__ == "__main__":
    main()
