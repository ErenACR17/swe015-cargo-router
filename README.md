# SWE015 Homework: Cargo Support Ticket Router (Use Case 2)

**Student:** Eren Acar (`2309111059`)
**Course:** SWE015 – Introduction to Large Language Models
**Use case:** Customer Support Ticket Router 

## Overview
A free-text customer message is normalized, tracking codes are extracted, spelling is corrected, and the intent is predicted with a confidence score. Low-confidence or `out_of_scope` messages go to a **Human Agent**; greetings get a rule-based (ELIZA-style) reply. Four models are shown side by side and the ticket is routed by the **quantized INT8 Transformer** (R5).

| Area | Component | Technique |
|---|---|---|
| B | `preprocessing.py` | Regex code extraction, text normalization, Levenshtein edit-distance spelling correction, ELIZA small-talk rules |
| B | Model 1 | TF-IDF + Logistic Regression (baseline) |
| A | Model 2 | Word2Vec (averaged) + 2-layer feed-forward NN, mini-batch SGD + momentum, dropout, backprop |
| C | Model 3 | `bert-base-uncased` fine-tuned for 6-class intent classification |
| D | Model 4 | Knowledge distillation into `distilbert-base-uncased` + dynamic INT8 quantization (`torch.ao.quantization.quantize_dynamic`, Linear layers) |

## Data
Self-generated, template-based dataset (`dataset.py`): 6 intents × 150 messages = 900 (train 612 / val 108 / test 180).
The **test set uses templates that never appear in train/val** (no leakage). Typos and noise are added randomly.
No external dataset is used; the 50–100 hand-written-example option of the assignment is replaced by templates.

## Setup & Run (R7)
```bash
pip install -r requirements.txt
```
**Trained models are not stored in this repo (too large).** Get them in one of two ways:

1. Download `models/` from: **<https://drive.google.com/file/d/1Vb5w7klxcQ-iSMebO-V6BqELudiIuoRr/view?usp=sharing>** and unzip it next to `app.py`, or
2. Train them yourself (≈ 5 min on a Colab T4 via `SWE015_Colab_Training.ipynb`, slower on CPU):
   ```bash
   python train_models.py      # creates models/ and results.json
   ```

Single run command:
```bash
python app.py
```
Gradio opens at `http://127.0.0.1:7860`.

## Results (R4)
Produced by `python train_models.py` → `results.json` (same held-out test set for all models, latency averaged over 100 runs on CPU, hardware string stored in `results.json`).
Paste your own measured table here after your final run.

| Model | Accuracy (%) | OOS recall (%) | Macro-F1 (%) | Size (MB) | Latency (ms) |
|---|---|---|---|---|---|
| 1. TF-IDF + Logistic Regression | 47.22 | 50.0 | 41.63 | 0.02 | 0.56 |
| 2. Word2Vec + SGD FFNN | 39.44 | 50.0 | 40.10 | 0.03 | 0.10 |
| 3. Fine-tuned BERT (FP32) | 90.56 | 100.0 | 90.06 | 418.36 | 21.98 |
| 4a. Distilled DistilBERT (FP32) [extra] | 91.67 | 100.0 | 91.11 | 255.44 | 11.02 |
| 4. Distilled + Quantized (INT8) | 85.00 | 100.0 | 84.17 | 132.30 | 13.92 |

Test set: 180 messages (unseen templates), same for all models. Latency: mean of 100 runs on one sample, CPU.
Hardware: Windows | Intel64 Family 6 Model 186 | 14 torch threads | torch 2.15.0+cpu | inference on CPU