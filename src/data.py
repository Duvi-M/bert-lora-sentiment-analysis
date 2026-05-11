from datasets import load_dataset
from transformers import AutoImageProcessor, AutoTokenizer


IMAGE_TASKS = {"image"}
TEXT_TASKS = {"sentiment", "topic"}


def prepare_dataset(config: dict):
    task_type = config.get("task_type", "sentiment")

    if task_type in TEXT_TASKS:
        return prepare_text_dataset(config)

    if task_type in IMAGE_TASKS:
        return prepare_image_dataset(config)

    supported_tasks = sorted(TEXT_TASKS | IMAGE_TASKS)
    raise ValueError(
        f"Unsupported task_type '{task_type}'. "
        f"Supported tasks are: {supported_tasks}."
    )


def prepare_text_dataset(config: dict):
    dataset_cfg = config["dataset"]
    model_name = config["model"]["name"]
    seed = config["seed"]
    text_column = dataset_cfg.get("text_column", "text")

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
            example[text_column],
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


def prepare_image_dataset(config: dict):
    dataset_cfg = config["dataset"]
    model_name = config["model"]["name"]
    seed = config["seed"]
    image_column = dataset_cfg.get("image_column", "image")

    dataset = load_dataset(dataset_cfg["name"])
    train_dataset = _select_split(dataset, "train", dataset_cfg["train_size"], seed)
    eval_split = "validation" if "validation" in dataset else "test"
    test_dataset = _select_split(dataset, eval_split, dataset_cfg["test_size"], seed)

    processor = AutoImageProcessor.from_pretrained(model_name)
    label_column = _resolve_label_column(train_dataset)

    def preprocess_images(examples):
        images = [
            image.convert("RGB")
            for image in examples[image_column]
        ]
        inputs = processor(images=images, return_tensors="pt")
        inputs["labels"] = examples[label_column]
        return inputs

    train_dataset = train_dataset.map(
        preprocess_images,
        batched=True,
        remove_columns=train_dataset.column_names,
    )
    test_dataset = test_dataset.map(
        preprocess_images,
        batched=True,
        remove_columns=test_dataset.column_names,
    )

    train_dataset.set_format(type="torch", columns=["pixel_values", "labels"])
    test_dataset.set_format(type="torch", columns=["pixel_values", "labels"])

    return train_dataset, test_dataset, processor


def _select_split(dataset, split_name: str, size: int, seed: int):
    split = dataset[split_name].shuffle(seed=seed)
    split_size = min(size, len(split))
    return split.select(range(split_size))


def _resolve_label_column(dataset):
    if "label" in dataset.column_names:
        return "label"

    if "labels" in dataset.column_names:
        return "labels"

    raise ValueError(
        "Could not find a label column. Expected 'label' or 'labels'."
    )
