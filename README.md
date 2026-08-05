# LLM-Prompt-Injection-Detector

## Highlights

- This code achieves a 99.1% accuracy on a held out test set in detecting benign vs malicious English LLM prompts via finetuning a BERT encoder
- Constructed a dataset containing over 300,000 English prompts by combining 3 separate prompt injection HuggingFace datasets
- Compare results against a scikit-learn logistic regression baseline showing a 3.4% increase in model accuracy
- Optimize benign/malicious decisions by tuning classification threshold to account for slight class imabalance (~55%-45% benign/malicious)

## Overview

Large Language Models (LLMs) have recently been at the forefront of technological innovation in the last 5 years. Their daily use has skyrocketed and has become a useful tool for the general population, not just researchers. However, they are not without their flaws. LLM’s are prone to hallucinations in their answers, and are extremely susceptible to bias during training, especially since they are often trained on the entire internet. Further, because of their capabilities they pose a significant risk when misused for malicious purposes. In a time where finding out public information of individuals has never been easier, protecting the public from attacks against their right to privacy becomes even more essential. That is where the scope of this project comes in. To protect LLMs against malicious prompt engineered attacks, a model is needed to discern which prompts are malicious, and which ones are benign. Using public datasets we train a simple fine-tuned BERT encoded model that can accurately detect attacks against LLMs in spite of various prompt engineering strategies, some of which may be unseen by our model during training. The main goal of this project is to aid researchers creating these LLMs to use as a tool during training to guide the model to not reveal sensitive information or return dangerous model outputs. 

## Authors
- Cooper Cox

## File Structure

## File Structure

```
LLM-Prompt-Injection-Detector/
├── data/                       # Data loading, curation, and dataset utilities
│   ├── dataset/                # Store curated dataset.csv here (not tracked by git)
│   ├── load_and_save.py        # Dataset curation and preprocessing pipeline
│   └── dataset.py              # Custom PyTorch Dataset class for fine-tuned BERT
├── logs/                       # Training logs and output files
├── model/                      # Model architecture and saved weights
│   ├── weights/                # Saved model weights (not tracked by git)
│   └── promptClassifier.py     # Custom prompt injection classifier model definition
├── train/                      # Training, evaluation, and model saving scripts
│   ├── train_bert.py           # Fine-tunes BERT classifier, evaluates, and saves weights
│   └── train_logreg.py         # Trains logistic regression baseline, evaluates, and saves weights
├── main.py                     # Entry point — run this to train/evaluate
├── sweep.yaml                  # WandB hyperparameter sweep configuration
├── pyproject.toml              # Project dependencies and metadata (managed by uv)
├── uv.lock                     # Locked dependency versions for reproducibility
└── README.md                   # Project documentation
```

## UV Installation
```
curl -LsSf https://astral.sh/uv/install.sh | sh

or

pip install uv
```

in the repo:
```
uv sync [optional flag] --no-cache
```

usage:
```
uv run main.py

or

source .venv/bin/activate
python3 main.py

uv add [python package]
```

## Dataset

- **[jayavibhav/prompt-injection-safety](https://huggingface.co/datasets/jayavibhav/prompt-injection-safety)** — 60,000 labeled prompts for prompt injection detection. No license specified; 

- **[jayavibhav/prompt-injection](https://huggingface.co/datasets/jayavibhav/prompt-injection/viewer)** — Prompt injection dataset by the same author. No license specified;

- **[rogue-security/prompt-injections-benchmark](https://huggingface.co/datasets/rogue-security/prompt-injections-benchmark)** — 5,000 prompts labeled as jailbreak or benign. Licensed under [CC-BY-NC-4.0](https://creativecommons.org/licenses/by-nc/4.0/) — free to use for non-commercial research with attribution.

## Usage Instructions

```bash
# Train a logistic regression model (default)
uv run main.py

# Train a BERT model
uv run main.py --model bert

# Evaluate on test split
uv run main.py --model bert --test True

# Create a new curated dataset
uv run main.py --dataset True

# Full example: train bert with custom hyperparameters and W&B logging
uv run main.py --model bert --epoch 20 --batch 16 --lr 0.00002 --threshold 0.7 --use_wandb
```

## Hyperparameters

| Flag | Short | Type | Default | Description |
|---|---|---|---|---|
| `--dataset` | `-D` | bool | `False` | Whether to curate and create a new dataset |
| `--model` | `-M` | str | `logreg` | Model architecture to use (`logreg`, `bert`, `large-bert`) |
| `--test` | `-t` | bool | `False` | Evaluate on test split instead of training |
| `--data_path` | `-d` | Path | `data/dataset/dataset.csv` | Path to the curated dataset CSV file |
| `--threshold` | `-th` | float | `0.5` | Classification threshold for predicting a malicious prompt |
| `--epoch` | `-e` | int | `10` | Number of training epochs |
| `--batch` | `-b` | int | `32` | Batch size (BERT models only) |
| `--seq_length` | `-seq` | int | `128` | Maximum token sequence length for prompt inputs |
| `--lr` | `-lr` | float | `0.00001` | Optimizer learning rate |
| `--save_path` | `-s` | Path | `model/weights/` | Directory to save model weights after training |
| `--use_wandb` | | flag | `False` | Enable Weights & Biases logging for the run |

## Results

### Overall Performance

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Logistic Regression | 0.957 | 0.968 | 0.951 | 0.959 |
| Fine-tuned BERT | 0.991 | 0.992 | 0.991 | 0.992 |

### Logistic Regression Threshold Analysis

| Metric | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 |
|---|---|---|---|---|---|---|---|---|---|
| Accuracy | 0.893 | 0.942 | 0.955 | 0.957 | 0.954 | 0.947 | 0.934 | 0.914 | 0.870 |
| Precision | 0.836 | 0.915 | 0.949 | 0.968 | 0.979 | 0.987 | 0.991 | 0.996 | 0.996 |
| Recall | 0.994 | 0.981 | 0.966 | 0.951 | 0.934 | 0.912 | 0.884 | 0.841 | 0.756 |

Note: Threshold analysis for the BERT classifier was not conducted due to the long training times (~10 hours/model run on full dataset)

## Relevant Papers

Papers that inspired this work:
- https://ieeexplore.ieee.org/document/10690068
- https://aclanthology.org/2025.findings-naacl.123.pdf
- https://arxiv.org/html/2602.16304v2
