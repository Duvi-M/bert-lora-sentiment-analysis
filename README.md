# DULoRA: Dynamic Utility-based LoRA Rank Allocation

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red)
![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Transformers-yellow)
![PEFT](https://img.shields.io/badge/PEFT-LoRA-orange)
![Research](https://img.shields.io/badge/Research-Master's%20Thesis-purple)

DULoRA is a master's thesis research pipeline for studying **dynamic utility-based rank allocation in LoRA-based fine-tuning**.

The project compares a standard fixed-rank LoRA baseline with an adaptive LoRA configuration where ranks are assigned per layer according to gradient-based utility scores. The main goal is to analyze whether LoRA rank capacity can be distributed more efficiently across transformer layers under a fixed rank budget.

The current implementation focuses on binary sentiment classification using transformer models and the IMDb dataset as the main benchmark. The pipeline is built with PyTorch, Hugging Face Transformers, Hugging Face Datasets, PEFT, and the Hugging Face `Trainer` API.

---

## Table of Contents

- [Research Motivation](#research-motivation)
- [Research Objective](#research-objective)
- [Method Overview](#method-overview)
- [Adaptive Rank Allocation Algorithm](#adaptive-rank-allocation-algorithm)
- [Pipeline Overview](#pipeline-overview)
- [Experimental Results](#experimental-results)
- [Generated Outputs](#generated-outputs)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Running an Experiment](#running-an-experiment)
- [Configuration](#configuration)
- [Reproducibility](#reproducibility)
- [Current Limitations](#current-limitations)
- [Future Work](#future-work)
- [Author](#author)

---

## Research Motivation

Low-Rank Adaptation, or LoRA, is a parameter-efficient fine-tuning technique that reduces the number of trainable parameters required to adapt large transformer models to downstream tasks.

In standard LoRA, the same rank value is usually applied to all selected target layers. However, not all layers necessarily contribute equally to the final task performance. Some layers may receive stronger optimization signals, while others may require less adaptation capacity.

DULoRA explores the hypothesis that LoRA rank should not always be distributed uniformly. Instead, rank capacity can be allocated according to layer utility, allowing more important layers to receive higher ranks while keeping less important layers at lower ranks.

The proposed approach estimates layer utility using gradient information collected during a short warmup stage and then distributes a fixed rank budget across LoRA target layers.

---

## Research Objective

The main objective of this project is to develop and evaluate an adaptive rank allocation method for LoRA-based fine-tuning of transformer models.

The project focuses on the following goals:

- Implement a reproducible LoRA fine-tuning pipeline for transformer-based text classification.
- Compare fixed-rank Baseline LoRA with Adaptive Gradient-Aware LoRA.
- Estimate layer utility using gradient norms from LoRA parameters.
- Allocate per-layer LoRA ranks under a configurable total rank budget.
- Save metrics, layer scores, rank patterns, allocation history, and plots for thesis analysis.
- Provide a clean experiment structure suitable for academic reporting and future extensions.

---

## Method Overview

The experiment pipeline follows three main stages.

### 1. Baseline LoRA

A standard LoRA model is trained using a fixed rank value defined in the configuration file.

This baseline is used as the reference point for evaluating the adaptive method.

Example:

```yaml
baseline_lora:
  r: 8
```

In this case, all selected LoRA target layers use rank `r = 8`.

---

### 2. Layer Utility Scoring

A temporary LoRA scoring model is created using the minimum adaptive rank. During a short warmup stage, the model processes a configurable number of batches and collects gradients from LoRA parameters.

The gradient norms are used as utility indicators for each LoRA layer.

The intuition is that layers with stronger gradient signals may benefit from receiving more rank capacity during the final adaptive training stage.

Example:

```yaml
adaptive_lora:
  min_rank: 2
  warmup_steps: 20
```

---

### 3. Adaptive LoRA

After computing layer utility scores, the pipeline creates a final adaptive LoRA model.

Instead of assigning the same rank to every layer, the model receives a `rank_pattern`, where each target layer can have a different rank.

The rank allocator starts with `min_rank` for all target layers and then increases the rank of selected layers according to their utility scores, while respecting the following constraints:

- Minimum rank per layer.
- Maximum rank per layer.
- Rank increment step.
- Total rank budget.

Example:

```yaml
adaptive_lora:
  min_rank: 2
  max_rank: 8
  rank_step: 2
  total_budget: 96
```

The core allocation logic is implemented in:

```text
src/rank_allocator.py
```

---

## Adaptive Rank Allocation Algorithm

DULoRA uses a gradient-aware rank allocation strategy.

Let each LoRA target layer be represented as:

```text
l = 1, 2, ..., L
```

Each layer receives a utility score:

```text
u_l
```

where `u_l` is computed from the gradient norms of LoRA parameters during the scoring stage.

The algorithm starts by assigning the minimum rank to each layer:

```text
r_l = r_min
```

Then, while the total budget allows further allocation, the algorithm selects the layer with the highest utility score that has not reached the maximum rank:

```text
r_l = r_l + rank_step
```

The process continues until one of the following conditions is met:

- The total rank budget is reached.
- All layers have reached the maximum rank.
- No further valid allocation is possible.

The final result is a layer-wise rank pattern:

```text
rank_pattern = {
  layer_1: r_1,
  layer_2: r_2,
  ...
  layer_L: r_L
}
```

This rank pattern is then passed to PEFT when building the adaptive LoRA model.

---

## Pipeline Overview

The complete experimental pipeline can be summarized as follows:

```text
IMDb Dataset
     ↓
Tokenization
     ↓
Baseline LoRA Training
     ↓
Gradient-based Layer Utility Scoring
     ↓
Adaptive Rank Allocation
     ↓
Adaptive LoRA Training
     ↓
Evaluation
     ↓
Metrics, CSV Files, and Plots
```

<!-- ESPAÑOL:
Aquí puedes agregar una imagen bonita del pipeline general.
Recomendación: crear una imagen llamada assets/pipeline_overview.png.

Ejemplo:

![DULoRA pipeline](assets/pipeline_overview.png)

La imagen debería mostrar algo como:
Dataset → Tokenization → Baseline LoRA → Gradient Scoring → Rank Allocation → Adaptive LoRA → Evaluation.
-->

---

## Experimental Results

<!-- ESPAÑOL:
Esta es una de las secciones más importantes.
Aquí debes colocar los resultados reales de tus experimentos.

Recomendación:
Crear una carpeta llamada assets/ y guardar ahí las imágenes importantes para que GitHub las muestre correctamente.

Ejemplo de estructura:

assets/
├── pipeline_overview.png
├── layer_scores_budget_96.png
├── rank_pattern_budget_96.png
├── budget_history_budget_96.png
├── accuracy_comparison.png
├── loss_comparison.png
└── trainable_params_comparison.png

No uses directamente outputs/run_001/... si esa carpeta está en .gitignore.
Mejor copia los gráficos finales o representativos a assets/.
-->

### Summary of Main Results

<!-- ESPAÑOL:
Aquí coloca una tabla con tus resultados reales.
Puedes usar los mejores resultados o los experimentos más representativos.

Ejemplo:
| Method | Accuracy | Eval Loss | Trainable Parameters | Training Time |
| --- | ---: | ---: | ---: | ---: |
| Baseline LoRA | 0.717 | 0.6601 | 294,912 | 164 s |
| DULoRA | 0.693 | 0.6652 | 294,912 | 172 s |

Si todavía no quieres poner números definitivos, puedes dejar la tabla como "To be updated".
-->

| Method | Accuracy | Eval Loss | Trainable Parameters | Training Time |
| --- | ---: | ---: | ---: | ---: |
| Baseline LoRA | <!-- ESPAÑOL: coloca accuracy real --> | <!-- ESPAÑOL: coloca loss real --> | <!-- ESPAÑOL: coloca parámetros entrenables --> | <!-- ESPAÑOL: coloca tiempo --> |
| DULoRA | <!-- ESPAÑOL: coloca accuracy real --> | <!-- ESPAÑOL: coloca loss real --> | <!-- ESPAÑOL: coloca parámetros entrenables --> | <!-- ESPAÑOL: coloca tiempo --> |

<!-- ESPAÑOL:
Aquí puedes escribir una interpretación breve y honesta.
Por ejemplo:

The current results show that the proposed method is functional and produces interpretable rank allocation patterns. However, in the current configuration, DULoRA does not consistently outperform the fixed-rank LoRA baseline. Further experiments with larger datasets, multiple seeds, and alternative utility functions are planned.

Si tus resultados mejoran, puedes cambiar esta interpretación.
-->

---

### Layer Utility Scores

This figure shows the normalized gradient-based utility score estimated for each LoRA target layer. Higher values indicate layers that received stronger gradient signals during the warmup stage.

<!-- ESPAÑOL:
Coloca aquí el gráfico de layer scores.

Ejemplo:
![Layer utility scores](assets/layer_scores_budget_96.png)
-->

![Layer utility scores](assets/layer_scores_budget_96.png)

---

### Adaptive Rank Distribution

This figure shows the final adaptive rank assigned to each LoRA layer after applying the rank allocation algorithm.

<!-- ESPAÑOL:
Coloca aquí el gráfico de rank pattern.

Ejemplo:
![Adaptive rank pattern](assets/rank_pattern_budget_96.png)
-->

![Adaptive rank pattern](assets/rank_pattern_budget_96.png)

---

### Rank Budget Allocation History

This figure shows how the rank budget is distributed across layers during the allocation process.

<!-- ESPAÑOL:
Coloca aquí el gráfico de allocation history o budget history.

Ejemplo:
![Budget allocation history](assets/budget_history_budget_96.png)
-->

![Budget allocation history](assets/budget_history_budget_96.png)

---

### Baseline LoRA vs DULoRA

<!-- ESPAÑOL:
Aquí coloca gráficos comparativos.
Recomendación:
1. accuracy_comparison.png
2. loss_comparison.png
3. trainable_params_comparison.png

Si solo tienes accuracy y loss, está bien.
-->

#### Accuracy Comparison

![Accuracy comparison](assets/accuracy_comparison.png)

#### Loss Comparison

![Loss comparison](assets/loss_comparison.png)

#### Trainable Parameter Comparison

![Trainable parameter comparison](assets/trainable_params_comparison.png)

---

## Generated Outputs

Each experiment run saves the following artifacts inside its output directory:

```text
outputs/<experiment_name>/
│
├── metrics.csv
├── layer_scores.csv
├── rank_pattern.csv
├── allocation_history.csv
│
├── layer_scores.png
├── rank_pattern.png
├── budget_history.png
│
├── baseline_lora/
└── adaptive_lora/
```

### Output Files

| File | Description |
| --- | --- |
| `metrics.csv` | Baseline and adaptive accuracy, loss, training time, and parameter counts. |
| `layer_scores.csv` | Normalized gradient-based utility score for each LoRA layer. |
| `rank_pattern.csv` | Final adaptive rank assigned to each target layer. |
| `allocation_history.csv` | Iteration-by-iteration record of rank budget allocation. |
| `layer_scores.png` | Bar plot of layer utility scores. |
| `rank_pattern.png` | Bar plot of assigned adaptive ranks. |
| `budget_history.png` | Plot of rank budget usage over allocation iterations. |
| `baseline_lora/` | Hugging Face `Trainer` output directory for Baseline LoRA. |
| `adaptive_lora/` | Hugging Face `Trainer` output directory for Adaptive LoRA. |

---

## Project Structure

```text
DULoRA-Dynamic-Utility-based-LoRA-Rank-Allocation/
│
├── assets/
│   ├── pipeline_overview.png
│   ├── layer_scores_budget_96.png
│   ├── rank_pattern_budget_96.png
│   ├── budget_history_budget_96.png
│   ├── accuracy_comparison.png
│   ├── loss_comparison.png
│   └── trainable_params_comparison.png
│
├── configs/
│   ├── default.yaml
│   ├── colab_budget_144.yaml
│   └── colab_budget_192.yaml
│
├── outputs/
│   └── .gitkeep
│
├── src/
│   ├── __init__.py
│   ├── data.py
│   ├── model.py
│   ├── rank_allocator.py
│   ├── experiment.py
│   ├── plots.py
│   ├── utils.py
│   └── evaluate.py
│
├── run_experiment.py
├── requirements.txt
├── README.md
└── .gitignore
```

<!-- ESPAÑOL:
Si todavía no tienes la carpeta assets/, créala.
Ahí debes colocar las imágenes finales que quieres mostrar en GitHub.

Comando:

mkdir assets
-->

---

## Main Components

| File | Purpose |
| --- | --- |
| `configs/default.yaml` | Default experiment configuration. |
| `configs/colab_budget_144.yaml` | Colab-oriented configuration with adaptive rank budget 144. |
| `configs/colab_budget_192.yaml` | Colab-oriented configuration with adaptive rank budget 192. |
| `src/data.py` | Loads and tokenizes the Hugging Face dataset. |
| `src/model.py` | Builds the base transformer model and applies LoRA through PEFT. |
| `src/rank_allocator.py` | Collects gradient-based layer scores and allocates adaptive ranks. |
| `src/experiment.py` | Orchestrates baseline training, scoring, adaptive training, and output saving. |
| `src/evaluate.py` | Defines evaluation metrics for Hugging Face `Trainer`. |
| `src/plots.py` | Saves CSV files and plots for analysis. |
| `src/utils.py` | Handles config loading, seed control, device selection, and parameter counting. |
| `run_experiment.py` | Command-line entrypoint for running experiments. |

---

## Installation

Create and activate a Python virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

The project depends on:

- PyTorch
- Hugging Face Transformers
- Hugging Face Datasets
- PEFT
- Accelerate
- Evaluate
- Scikit-learn
- Matplotlib
- Pandas
- PyYAML

<!-- ESPAÑOL:
Opcionalmente puedes agregar aquí la versión de Python recomendada.
Por ejemplo:

This project was tested with Python 3.10.

Si trabajaste en Colab, también puedes poner:

The experiments were mainly executed in Google Colab using GPU acceleration.
-->

---

## Running an Experiment

Run the default experiment:

```bash
python run_experiment.py --config configs/default.yaml
```

Run a specific budget configuration:

```bash
python run_experiment.py --config configs/colab_budget_144.yaml
```

```bash
python run_experiment.py --config configs/colab_budget_192.yaml
```

---

## Reproducing Thesis Experiments

<!-- ESPAÑOL:
Aquí puedes poner los comandos exactos que usaste para los experimentos de tu tesis.

Si tienes varios presupuestos, puedes colocar algo así.
Modifica los nombres según tus archivos reales.
-->

To reproduce the main thesis experiments, run:

```bash
python run_experiment.py --config configs/default.yaml
python run_experiment.py --config configs/colab_budget_144.yaml
python run_experiment.py --config configs/colab_budget_192.yaml
```

The generated metrics and plots will be saved under:

```text
outputs/<experiment_name>/
```

### Fair budget comparison experiments

These configurations compare Adaptive LoRA against fixed-rank LoRA under
approximately equivalent rank budgets:

- Fixed LoRA `r=4` vs Adaptive LoRA budget `96`.
- Fixed LoRA `r=6` vs Adaptive LoRA budget `144`.
- Fixed LoRA `r=8` vs Adaptive LoRA budget `192`.

Run the fair-comparison experiments with:

```bash
python run_experiment.py --config configs/fair_budget_96.yaml
python run_experiment.py --config configs/fair_budget_144.yaml
python run_experiment.py --config configs/fair_budget_192.yaml
```

### Final thesis plots

After running the fair-comparison experiments, generate the final thesis-ready
plots with:

```bash
python generate_final_plots.py
```

The script reads `outputs/fair_budget_96`, `outputs/fair_budget_144`, and
`outputs/fair_budget_192`, then saves the consolidated figures and summary CSV
under:

```text
outputs/final_plots/
```

<!-- ESPAÑOL:
Si luego haces experimentos con varias seeds, agrega algo así:

python run_experiment.py --config configs/seed_42_budget_96.yaml
python run_experiment.py --config configs/seed_96_budget_96.yaml
python run_experiment.py --config configs/seed_123_budget_96.yaml
-->

---

## Configuration

The default configuration follows this structure:

```yaml
seed: 42
experiment_name: run_001
output_dir: outputs

dataset:
  name: imdb
  train_size: 2000
  test_size: 1000
  max_length: 256

model:
  name: bert-base-uncased
  num_labels: 2

training:
  epochs: 2
  batch_size: 8
  learning_rate: 0.00005
  logging_steps: 50

lora:
  alpha: 16
  dropout: 0.1
  target_modules:
    - query
    - value

baseline_lora:
  r: 8

adaptive_lora:
  algorithm: gradient_aware
  min_rank: 2
  max_rank: 8
  rank_step: 2
  total_budget: 96
  warmup_steps: 20
```

### Configuration Fields

| Field | Description |
| --- | --- |
| `seed` | Random seed used for reproducibility. |
| `experiment_name` | Name of the output folder for the experiment. |
| `output_dir` | Root directory where experiment outputs are saved. |
| `dataset.name` | Hugging Face dataset name. |
| `dataset.train_size` | Number of training samples used. |
| `dataset.test_size` | Number of evaluation samples used. |
| `dataset.max_length` | Maximum tokenized sequence length. |
| `model.name` | Pretrained transformer model name. |
| `model.num_labels` | Number of output labels. |
| `training.epochs` | Number of training epochs. |
| `training.batch_size` | Training and evaluation batch size. |
| `training.learning_rate` | Optimizer learning rate. |
| `lora.alpha` | LoRA scaling factor. |
| `lora.dropout` | LoRA dropout value. |
| `lora.target_modules` | Transformer modules where LoRA is applied. |
| `baseline_lora.r` | Fixed LoRA rank used by the baseline model. |
| `adaptive_lora.min_rank` | Minimum rank assigned to each adaptive LoRA layer. |
| `adaptive_lora.max_rank` | Maximum rank allowed for each adaptive LoRA layer. |
| `adaptive_lora.rank_step` | Increment used when increasing a layer rank. |
| `adaptive_lora.total_budget` | Total rank budget available for adaptive allocation. |
| `adaptive_lora.warmup_steps` | Number of batches used for gradient-based scoring. |

---

## Output Directory Behavior

Experiments are saved in a dedicated run directory.

If `experiment_name` is provided:

```yaml
experiment_name: budget_144
output_dir: outputs
```

the run is saved to:

```text
outputs/budget_144/
```

If `experiment_name` is omitted, the pipeline creates a timestamped folder:

```text
outputs/run_2026-05-08_15-30-20/
```

This prevents results from different runs from being accidentally overwritten.

---

## Reproducibility

The pipeline uses the configured `seed` to improve reproducibility:

- Dataset shuffling uses the configured seed.
- Baseline, scoring, and adaptive model initialization reset the seed before model construction.
- Hugging Face `TrainingArguments` receives both `seed` and `data_seed`.
- Output folders are isolated by `experiment_name` or timestamp.

Exact reproducibility may still depend on hardware, backend behavior, CUDA/cuDNN settings, and library versions.

<!-- ESPAÑOL:
Si quieres ser más preciso, puedes agregar versiones reales de tus librerías.
Por ejemplo:

The main experiments were executed with:
- Python 3.10
- PyTorch ...
- Transformers ...
- PEFT ...

Pero solo ponlo si estás seguro de las versiones.
-->

---

## Current Evaluation

The current evaluation reports:

- Accuracy
- Evaluation loss
- Training time
- Trainable parameter count
- Total parameter count

These values are returned by the experiment pipeline and saved to:

```text
metrics.csv
```

---

## Example Research Questions

This repository is designed to support the following research questions:

- Can gradient-based utility scores identify LoRA layers that benefit from higher rank?
- Does adaptive rank allocation improve performance under the same or similar trainable parameter budget?
- How does the total rank budget affect accuracy, loss, and parameter efficiency?
- Are rank allocations stable across different seeds, dataset sizes, or model architectures?
- Which transformer layers tend to receive higher adaptive ranks during fine-tuning?

---

## Current Limitations

The current implementation has several limitations:

- The main benchmark is currently IMDb binary sentiment classification.
- The adaptive allocation is performed before final training and is not updated dynamically during training.
- The current utility function is based on gradient norms only.
- Results may vary across random seeds, hardware, and library versions.
- More experiments are needed to validate the method across additional datasets and model architectures.
- The current comparison focuses mainly on accuracy, loss, training time, and parameter count.

These limitations are part of the ongoing research process and provide directions for future improvements.

---

## Future Work

Possible extensions include:

- Testing additional datasets beyond IMDb.
- Evaluating other transformer backbones such as RoBERTa, DistilBERT, or DeBERTa.
- Extending the method to topic classification datasets such as AG News.
- Extending the method to image classification with Vision Transformers.
- Running experiments with multiple random seeds.
- Adding confidence intervals and statistical analysis.
- Comparing different utility functions for rank allocation.
- Updating rank allocation dynamically during training.
- Logging experiments with Weights & Biases or TensorBoard.
- Extending evaluation metrics beyond accuracy.
- Adding automated experiment tables for thesis reporting.
- Creating a more detailed ablation study for rank budget, warmup steps, and rank constraints.

---

## Academic Context

This repository was developed as part of a master's thesis project on parameter-efficient fine-tuning of transformer models.

The research focuses on adaptive rank allocation in LoRA and investigates whether layer-wise rank distribution can improve the efficiency of transformer fine-tuning under constrained parameter budgets.

<!-- ESPAÑOL:
Aquí puedes agregar el título exacto de tu tesis en inglés o ruso.

Por ejemplo:

Master's thesis topic:
"Development of an Adaptive Rank Allocation Algorithm in the Low-Rank Adaptation Method for Efficient Fine-Tuning of Transformer Models"

O en ruso:

«Разработка алгоритма адаптивного распределения ранга в методе Low-Rank Adaptation для эффективной настройки трансформерных моделей»
-->

---

## Citation

<!-- ESPAÑOL:
Esto es opcional.
Puedes dejarlo si quieres que el repo se vea más académico.
Si todavía no quieres usarlo, puedes eliminar esta sección.
-->

If you use this repository or refer to this work, please cite it as:

```bibtex
@mastersthesis{mendoza2026dulora,
  title={Dynamic Utility-based LoRA Rank Allocation for Efficient Transformer Fine-Tuning},
  author={Mendoza, Duvan},
  school={MEPhI - Moscow Engineering Physics Institute},
  year={2026}
}
```

---

## Author

Duvan Mendoza  
MSc Software Engineering and Big Data  
MEPhI - Moscow Engineering Physics Institute

Research focus:

- Machine Learning
- Natural Language Processing
- Transformer Models
- Parameter-Efficient Fine-Tuning
- LoRA Rank Allocation
