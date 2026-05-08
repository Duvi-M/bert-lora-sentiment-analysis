from datasets import load_dataset
from transformers import AutoTokenizer


def prepare_dataset(config: dict):
    dataset_cfg = config["dataset"]
    model_name = config["model"]["name"]
    seed = config["seed"]

    dataset = load_dataset(dataset_cfg["name"])

    train_dataset = dataset["train"].shuffle(seed=seed).select(
        range(dataset_cfg["train_size"])
    )
    test_dataset = dataset["test"].shuffle(seed=seed).select(
        range(dataset_cfg["test_size"])
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize(example):
        return tokenizer(
            example["text"],
            truncation=True,
            padding="max_length",
            max_length=dataset_cfg["max_length"],
        )

    train_dataset = train_dataset.map(tokenize, batched=True)
    test_dataset = test_dataset.map(tokenize, batched=True)

    train_dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "label"],
    )
    test_dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "label"],
    )

    return train_dataset, test_dataset, tokenizer