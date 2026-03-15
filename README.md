# Sentiment Analysis with BERT and LoRA

This project explores **sentiment classification using transformer models**, focusing on the comparison between **full fine-tuning and parameter-efficient fine-tuning with LoRA (Low-Rank Adaptation)**.

The goal is to evaluate how LoRA can significantly reduce the number of trainable parameters while maintaining competitive performance.

This project is closely related to research in **efficient adaptation of large transformer models**, and demonstrates practical applications of modern NLP techniques using the Hugging Face ecosystem.

---

# Project Goals

The main objectives of this project are:

- Build a **sentiment classification system** using transformer-based models.
- Fine-tune **BERT** on the IMDb movie reviews dataset.
- Implement **LoRA (Low-Rank Adaptation)** for parameter-efficient fine-tuning.
- Compare **full fine-tuning vs LoRA adaptation**.
- Evaluate model performance using standard NLP metrics.

---

# Dataset

We use the **IMDb movie reviews dataset**, a standard benchmark for sentiment analysis.

Dataset characteristics:

- 50,000 movie reviews
- Binary sentiment classification
- Labels:
  - `0` → Negative review
  - `1` → Positive review

Dataset source:

https://huggingface.co/datasets/imdb

---

# Models

This project evaluates two training approaches:

## 1. Full Fine-Tuning

Standard transformer fine-tuning where **all model parameters are updated** during training.

Model:

