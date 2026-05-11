from transformers import (
    AutoModelForImageClassification,
    AutoModelForSequenceClassification,
)
from peft import LoraConfig, get_peft_model


IMAGE_TASKS = {"image"}
TEXT_TASKS = {"sentiment", "topic"}


def load_base_model(config: dict):
    task_type = config.get("task_type", "sentiment")
    model_name = config["model"]["name"]
    num_labels = config["model"]["num_labels"]

    if task_type in TEXT_TASKS:
        return AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
        )

    if task_type in IMAGE_TASKS:
        return AutoModelForImageClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            ignore_mismatched_sizes=True,
        )

    supported_tasks = sorted(TEXT_TASKS | IMAGE_TASKS)
    raise ValueError(
        f"Unsupported task_type '{task_type}'. "
        f"Supported tasks are: {supported_tasks}."
    )


def build_lora_model(config: dict, rank_pattern=None, r=None):
    base_model = load_base_model(config)

    if r is None:
        r = config["baseline_lora"]["r"]

    lora_cfg = config["lora"]

    lora_config_kwargs = {
        "r": r,
        "lora_alpha": lora_cfg["alpha"],
        "target_modules": lora_cfg["target_modules"],
        "lora_dropout": lora_cfg["dropout"],
        "bias": "none",
    }

    # Importante:
    # Solo agregamos rank_pattern cuando existe.
    # Para LoRA estándar NO debe pasarse rank_pattern=None.
    if rank_pattern is not None:
        lora_config_kwargs["rank_pattern"] = rank_pattern

    lora_config = LoraConfig(**lora_config_kwargs)

    return get_peft_model(base_model, lora_config)


def list_lora_target_module_candidates(config: dict, keywords=("query", "value")):
    model = load_base_model(config)
    return [
        name
        for name, _ in model.named_modules()
        if any(keyword in name for keyword in keywords)
    ]
