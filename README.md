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

Model: bert-base-uncased


## 2. Parameter-Efficient Fine-Tuning (LoRA)

Instead of updating all parameters, **LoRA introduces small trainable low-rank matrices** inside attention layers.

Advantages:

- drastically fewer trainable parameters
- faster training
- lower memory usage
- similar performance

Library used: PEFT (Parameter Efficient Fine-Tuning)


---

# Technology Stack

Main tools used in this project:

- Python
- PyTorch
- Hugging Face Transformers
- Hugging Face Datasets
- PEFT (LoRA)
- Scikit-learn
- TensorBoard / Weights & Biases (optional)

---


# Project Structure

```text
bert-lora-sentiment-analysis
│
├── data/ # dataset storage
├── notebooks/ # exploration and visualization
│
├── src/
│ ├── data.py # dataset loading
│ ├── model.py # model definition
│ ├── train.py # training pipeline
│ ├── evaluate.py # evaluation metrics
│ └── utils.py # helper functions
│
├── configs/ # experiment configurations
├── experiments/ # experiment outputs
├── results/ # evaluation results
│
├── requirements.txt
├── README.md
└── .gitignore

```
---

# Training Pipeline

The training process follows these steps:

1. Load the IMDb dataset
2. Tokenize text using a BERT tokenizer
3. Train baseline BERT with full fine-tuning
4. Train BERT with LoRA adaptation
5. Evaluate both models
6. Compare performance and efficiency

---

# Evaluation Metrics

Models are evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

Additional analysis:

- number of trainable parameters
- training efficiency
- model performance comparison

---

# Example Results

Example comparison (expected):

| Model | Trainable Parameters | Accuracy | F1 Score |
|------|------|------|------|
| BERT Fine-Tuning | 110M | ~0.94 | ~0.94 |
| BERT + LoRA | <1M | ~0.93 | ~0.93 |

LoRA maintains competitive performance while dramatically reducing the number of trainable parameters.

---

# Running the Project

Install dependencies: pip install -r requirements.txt

Train the baseline model: python src/train.py


Evaluate the model: python src/evaluate.py


---

# Future Improvements

Possible extensions of this project include:

- testing other transformer models (RoBERTa, DistilBERT)
- hyperparameter optimization
- training efficiency analysis
- experiment tracking with Weights & Biases
- deployment of the trained model as an API

---

# Author

Duvan Mendoza  
MSc Software Engineering & Big Data  
MEPhI (Moscow Engineering Physics Institute)

Focus areas:

- Machine Learning
- NLP
- Transformer Models
- Parameter-Efficient Fine-Tuning




