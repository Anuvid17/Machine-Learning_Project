# Sarcasm Detection Using BERT

A fine-tuned BERT model for binary sarcasm classification on Reddit comments.

---

## Overview

This project fine-tunes `bert-base-uncased` to detect sarcasm in text. Given a Reddit comment, the model predicts whether it is **sarcastic (1)** or **non-sarcastic (0)**.

---

## Model Architecture

| Component | Details |
|---|---|
| Base Model | `bert-base-uncased` |
| Task | Binary Sequence Classification |
| Framework | HuggingFace Transformers + PyTorch |
| Parameters | ~110 Million |
| Max Token Length | 128 |

---

## Project Structure
sarcasm-bert/
├── Final_Sarcasm_MLTP.ipynb      # Main notebook
├── modified_sarcasm_dataset.csv  # Dataset
├── results/                      # Saved model checkpoints
├── logs/                         # Training logs
└── README.md
---

## Dataset

- **File:** `modified_sarcasm_dataset.csv`
- **Source:** Modified Sarcasm on Reddit dataset
- **Columns:**
  - `comment` — raw Reddit comment text
  - `label` — `0` (non-sarcastic) / `1` (sarcastic)
- **Split:** 80% train / 20% test (`random_state=42`)

---

## Setup & Installation

```bash
pip install torch transformers pandas scikit-learn
```

> Verify GPU availability:
> ```bash
> python -c "import torch; print(torch.cuda.is_available())"
> ```

---

## How to Run

1. Clone the repo and place `modified_sarcasm_dataset.csv` in the root directory.
2. Open `Final_Sarcasm_MLTP.ipynb` in Jupyter or Google Colab.
3. Run all cells top to bottom.

The notebook will:
- Load and split the dataset
- Tokenise using `BertTokenizer`
- Fine-tune BERT for 2 epochs
- Print accuracy and a full classification report

---

## Training Configuration

| Hyperparameter | Value |
|---|---|
| Epochs | 2 |
| Train Batch Size | 8 |
| Eval Batch Size | 8 |
| Max Sequence Length | 128 |
| Optimiser | AdamW (HuggingFace default) |

---

## Evaluation

After training, the model is evaluated on the held-out test set:

```python
preds = trainer.predict(test_dataset)
y_pred = preds.predictions.argmax(axis=1)

print("Accuracy:", accuracy_score(test_labels, y_pred))
print(classification_report(test_labels, y_pred))
```

Metrics reported: **Accuracy, Precision, Recall, F1-Score**

---

## Estimated Runtime

| Hardware | Time (2 Epochs) |
|---|---|
| NVIDIA T4 / V100 GPU | 10–25 min |
| Apple M1/M2 (MPS) | 25–50 min |
| CPU only | 2–6 hours |

---

## Future Improvements

- [ ] Train for more epochs with early stopping
- [ ] Add `eval_strategy="epoch"` to monitor validation loss
- [ ] Try RoBERTa or DeBERTa for better performance
- [ ] Export model to ONNX for deployment
- [ ] Add cross-validation for more robust evaluation

---
