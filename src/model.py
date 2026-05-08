from transformers import AutoModelForSequenceClassification
from peft import LoraConfig, get_peft_model


def load_base_model(config: dict):
    return AutoModelForSequenceClassification.from_pretrained(
        config["model"]["name"],
        num_labels=config["model"]["num_labels"],
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