import time
from torch.utils.data import DataLoader
from transformers import TrainingArguments, Trainer

from src.data import prepare_dataset
from src.model import build_lora_model
from src.evaluate import build_compute_metrics
from src.rank_allocator import (
    collect_gradient_layer_scores,
    normalize_scores,
    allocate_ranks,
)
from src.utils import (
    get_device,
    count_trainable_params,
    count_total_params,
)
from src.plots import (
    save_layer_scores_plot,
    save_rank_pattern_plot,
    save_allocation_history_plot,
    save_metrics,
)


def train_and_evaluate(model, train_dataset, test_dataset, config, output_dir):
    training_cfg = config["training"]

    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=training_cfg["batch_size"],
        per_device_eval_batch_size=training_cfg["batch_size"],
        num_train_epochs=training_cfg["epochs"],
        learning_rate=training_cfg["learning_rate"],
        eval_strategy="epoch",
        logging_steps=training_cfg["logging_steps"],
        save_strategy="no",
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=build_compute_metrics(),
    )

    start = time.time()
    trainer.train()
    train_time = time.time() - start

    eval_results = trainer.evaluate()

    return eval_results, train_time


def run_experiment(config: dict):
    output_dir = config["output_dir"]
    device = get_device()

    train_dataset, test_dataset, tokenizer = prepare_dataset(config)

    # 1. Baseline LoRA
    baseline_model = build_lora_model(
        config=config,
        r=config["baseline_lora"]["r"],
    )

    baseline_results, baseline_time = train_and_evaluate(
        model=baseline_model,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        config=config,
        output_dir=f"{output_dir}/baseline_lora",
    )

    baseline_metrics = {
        "method": "baseline_lora",
        "accuracy": baseline_results.get("eval_accuracy"),
        "loss": baseline_results.get("eval_loss"),
        "train_time": baseline_time,
        "trainable_params": count_trainable_params(baseline_model),
        "total_params": count_total_params(baseline_model),
    }

    # 2. Scoring model para calcular gradientes
    adaptive_cfg = config["adaptive_lora"]

    scoring_model = build_lora_model(
        config=config,
        r=adaptive_cfg["min_rank"],
    )

    scoring_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
    )

    layer_scores = collect_gradient_layer_scores(
        model=scoring_model,
        dataloader=scoring_loader,
        max_steps=adaptive_cfg["warmup_steps"],
        device=device,
    )

    layer_scores = normalize_scores(layer_scores)

    rank_pattern, history = allocate_ranks(
        layer_scores=layer_scores,
        total_budget=adaptive_cfg["total_budget"],
        min_rank=adaptive_cfg["min_rank"],
        max_rank=adaptive_cfg["max_rank"],
        step=adaptive_cfg["rank_step"],
    )

    save_layer_scores_plot(layer_scores, output_dir)
    save_rank_pattern_plot(rank_pattern, output_dir)
    save_allocation_history_plot(history, output_dir)

    # 3. Adaptive Gradient-Aware LoRA
    adaptive_model = build_lora_model(
        config=config,
        rank_pattern=rank_pattern,
        r=adaptive_cfg["max_rank"],
    )

    adaptive_results, adaptive_time = train_and_evaluate(
        model=adaptive_model,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        config=config,
        output_dir=f"{output_dir}/adaptive_lora",
    )

    adaptive_metrics = {
        "method": "adaptive_gradient_lora",
        "accuracy": adaptive_results.get("eval_accuracy"),
        "loss": adaptive_results.get("eval_loss"),
        "train_time": adaptive_time,
        "trainable_params": count_trainable_params(adaptive_model),
        "total_params": count_total_params(adaptive_model),
    }

    save_metrics(
        {
            "baseline_accuracy": baseline_metrics["accuracy"],
            "baseline_loss": baseline_metrics["loss"],
            "baseline_trainable_params": baseline_metrics["trainable_params"],
            "adaptive_accuracy": adaptive_metrics["accuracy"],
            "adaptive_loss": adaptive_metrics["loss"],
            "adaptive_trainable_params": adaptive_metrics["trainable_params"],
        },
        output_dir,
    )

    return {
        "baseline": baseline_metrics,
        "adaptive": adaptive_metrics,
        "rank_pattern": rank_pattern,
        "layer_scores": layer_scores,
        "allocation_history": history,
    }