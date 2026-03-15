from datasets import load_dataset
from transformers import AutoTokenizer


def load_imdb_dataset():
    """
    Load the IMDb dataset from Hugging Face.

    Returns:
        DatasetDict: Hugging Face dataset with 'train' and 'test' splits.
    """
    dataset = load_dataset("imdb")
    return dataset


def get_tokenizer(model_name: str = "bert-base-uncased"):
    """
    Load tokenizer for the selected transformer model.

    Args:
        model_name (str): Hugging Face model name.

    Returns:
        AutoTokenizer: tokenizer instance.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer


def tokenize_function(examples, tokenizer, max_length: int = 256):
    """
    Tokenize a batch of text examples.

    Args:
        examples (dict): Batch of examples containing 'text'.
        tokenizer: Hugging Face tokenizer.
        max_length (int): Maximum sequence length.

    Returns:
        dict: Tokenized batch.
    """
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=max_length,
    )


def prepare_dataset(
    model_name: str = "bert-base-uncased",
    max_length: int = 256,
):
    """
    Load and tokenize IMDb dataset, then prepare it for training.

    Args:
        model_name (str): Hugging Face model name.
        max_length (int): Maximum token length.

    Returns:
        tuple: (train_dataset, test_dataset, tokenizer)
    """
    dataset = load_imdb_dataset()
    tokenizer = get_tokenizer(model_name)

    tokenized_dataset = dataset.map(
        lambda examples: tokenize_function(examples, tokenizer, max_length),
        batched=True,
    )

    tokenized_dataset = tokenized_dataset.remove_columns(["text"])
    tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
    tokenized_dataset.set_format("torch")

    train_dataset = tokenized_dataset["train"]
    test_dataset = tokenized_dataset["test"]

    return train_dataset, test_dataset, tokenizer


if __name__ == "__main__":
    train_dataset, test_dataset, tokenizer = prepare_dataset()

    print("Train dataset size:", len(train_dataset))
    print("Test dataset size:", len(test_dataset))
    print("Tokenizer loaded:", tokenizer.name_or_path)
    print("Sample item keys:", train_dataset[0].keys())