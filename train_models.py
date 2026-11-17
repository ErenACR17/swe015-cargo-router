"""
SWE015 Homework - Training & Evaluation Script for the 4-Model Ladder:
1. Model 1 (Classical Baseline - Area B): TF-IDF + Logistic Regression
2. Model 2 (Neural Network - Area A): Word2Vec + Feedforward NN trained with SGD & Backprop
3. Model 3 (Transformer - Area C): Fine-Tuned BERT (bert-base-uncased)
4. Model 4 (Quantized Transformer - Area D): Distilled Student (distilbert-base-uncased) + Dynamic INT8 Quantization
"""

import json
import os
import pickle
import platform
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, recall_score
from gensim.models import Word2Vec
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from dataset import INTENT_NAMES, load_dataset
from preprocessing import preprocess_message

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- Area A: Simple Feedforward Neural Network ---
class SimpleFFNN(nn.Module):
    def __init__(self, input_dim=100, hidden_dim=64, num_classes=6):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


def sentence_vector(tokens, w2v_model, dim=100):
    vecs = [w2v_model.wv[w] for w in tokens if w in w2v_model.wv]
    if len(vecs) == 0:
        return np.zeros(dim, dtype=np.float32)
    return np.mean(vecs, axis=0).astype(np.float32)


# --- Simple PyTorch Dataset for BERT / DistilBERT ---
class TicketDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=48):
        self.encodings = tokenizer(texts, truncation=True, padding="max_length", max_length=max_len, return_tensors="pt")
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels": self.labels[idx]
        }


def measure_metrics_and_latency(predict_fn, test_texts, y_true, model_file_path, n_runs=100):
    preds = [predict_fn(t) for t in test_texts]
    acc = accuracy_score(y_true, preds) * 100
    f1 = f1_score(y_true, preds, average="macro") * 100

    # Out-of-scope class is index 5
    oos_id = INTENT_NAMES.index("out_of_scope")
    oos_true = [1 if y == oos_id else 0 for y in y_true]
    oos_pred = [1 if p == oos_id else 0 for p in preds]
    oos_recall = recall_score(oos_true, oos_pred, zero_division=0) * 100

    # Measure average latency over 100 runs on a sample ticket
    sample_text = test_texts[0]
    for _ in range(5):
        predict_fn(sample_text)
    start = time.perf_counter()
    for _ in range(n_runs):
        predict_fn(sample_text)
    avg_latency_ms = ((time.perf_counter() - start) / n_runs) * 1000

    # Measure size on disk in MB
    if os.path.isdir(model_file_path):
        total_bytes = sum(f.stat().st_size for f in Path(model_file_path).glob("**/*") if f.is_file())
    else:
        total_bytes = os.path.getsize(model_file_path)
    size_mb = total_bytes / (1024 * 1024)

    return round(acc, 2), round(oos_recall, 2), round(f1, 2), round(size_mb, 2), round(avg_latency_ms, 2)


def hardware_description():
    """CPU model + threads + library versions (R4 asks to state the hardware)."""
    cpu = platform.processor()
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    cpu = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    return (f"{platform.system()} | CPU: {cpu or 'unknown'} | torch threads: {torch.get_num_threads()} | "
            f"torch {torch.__version__} | inference on CPU")


def main():
    print(f"Using device: {DEVICE}")
    train_raw, val_raw, test_raw = load_dataset(seed=42)

    train_prep = [preprocess_message(x["text"]) for x in train_raw]
    test_prep = [preprocess_message(x["text"]) for x in test_raw]

    X_train_text = [x["clean_text"] for x in train_prep]
    X_train_tokens = [x["tokens"] for x in train_prep]
    y_train = [x["label"] for x in train_raw]

    X_test_text = [x["clean_text"] for x in test_prep]
    X_test_tokens = [x["tokens"] for x in test_prep]
    y_test = [x["label"] for x in test_raw]

    # =========================================================================
    # 1. Model 1: Classical Baseline (TF-IDF + Logistic Regression)
    # =========================================================================
    print("Training Model 1: TF-IDF + Logistic Regression...")
    vectorizer = TfidfVectorizer(max_features=1000)
    X_tr_vec = vectorizer.fit_transform(X_train_text)
    clf = LogisticRegression(max_iter=300)
    clf.fit(X_tr_vec, y_train)

    tfidf_path = MODELS_DIR / "tfidf_baseline.pkl"
    with open(tfidf_path, "wb") as f:
        pickle.dump({"vectorizer": vectorizer, "clf": clf}, f)

    val_prep = [preprocess_message(x["text"]) for x in val_raw]
    val_acc = accuracy_score([x["label"] for x in val_raw], clf.predict(vectorizer.transform([x["clean_text"] for x in val_prep]))) * 100
    print(f"Model 1 validation accuracy: {val_acc:.2f}%")

    # =========================================================================
    # 2. Model 2: Word2Vec + Feedforward Neural Network (SGD & Backprop)
    # =========================================================================
    print("Training Model 2: Word2Vec + SGD Feedforward Neural Network...")
    w2v = Word2Vec(sentences=X_train_tokens, vector_size=100, window=4, min_count=1, epochs=25, seed=42)
    w2v_path = MODELS_DIR / "word2vec.model"
    w2v.save(str(w2v_path))

    X_tr_w2v = torch.tensor(np.array([sentence_vector(t, w2v) for t in X_train_tokens]), dtype=torch.float32)
    y_tr_tensor = torch.tensor(y_train, dtype=torch.long)

    ffnn = SimpleFFNN(input_dim=100, hidden_dim=64, num_classes=len(INTENT_NAMES))
    optimizer_sgd = torch.optim.SGD(ffnn.parameters(), lr=0.05, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    # Mini-batch SGD (batch size 32) with momentum + backpropagation
    torch.manual_seed(42)
    ffnn.train()
    n_tr = X_tr_w2v.shape[0]
    for epoch in range(40):
        perm = torch.randperm(n_tr)
        for start in range(0, n_tr, 32):
            idx = perm[start:start + 32]
            optimizer_sgd.zero_grad()
            loss = criterion(ffnn(X_tr_w2v[idx]), y_tr_tensor[idx])
            loss.backward()
            optimizer_sgd.step()

    ffnn.eval()
    ffnn_path = MODELS_DIR / "ffnn_sgd.pt"
    torch.save(ffnn.state_dict(), ffnn_path)

    # =========================================================================
    # 3. Model 3: Fine-Tuned BERT (Teacher Model)
    # =========================================================================
    print("Training Model 3: Fine-Tuning BERT (bert-base-uncased)...")
    teacher_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(teacher_name)
    teacher_model = AutoModelForSequenceClassification.from_pretrained(teacher_name, num_labels=len(INTENT_NAMES)).to(DEVICE)

    train_ds = TicketDataset(X_train_text, y_train, tokenizer)
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    opt_bert = torch.optim.AdamW(teacher_model.parameters(), lr=3e-5)

    teacher_model.train()
    for epoch in range(3):
        for batch in train_loader:
            opt_bert.zero_grad()
            inp = batch["input_ids"].to(DEVICE)
            mask = batch["attention_mask"].to(DEVICE)
            lbl = batch["labels"].to(DEVICE)
            outputs = teacher_model(input_ids=inp, attention_mask=mask, labels=lbl)
            outputs.loss.backward()
            opt_bert.step()

    teacher_model.eval()
    teacher_dir = MODELS_DIR / "bert_teacher"
    teacher_model.save_pretrained(teacher_dir)
    tokenizer.save_pretrained(teacher_dir)

    # =========================================================================
    # 4. Model 4: Knowledge Distillation + Dynamic INT8 Quantization
    # =========================================================================
    print("Training Model 4: Knowledge Distillation into DistilBERT + INT8 Quantization...")
    student_name = "distilbert-base-uncased"
    student_model = AutoModelForSequenceClassification.from_pretrained(student_name, num_labels=len(INTENT_NAMES)).to(DEVICE)
    opt_student = torch.optim.AdamW(student_model.parameters(), lr=5e-5)
    kl_loss = nn.KLDivLoss(reduction="batchmean")
    temp = 2.5
    alpha = 0.6

    student_model.train()
    for epoch in range(3):
        for batch in train_loader:
            opt_student.zero_grad()
            inp = batch["input_ids"].to(DEVICE)
            mask = batch["attention_mask"].to(DEVICE)
            lbl = batch["labels"].to(DEVICE)

            with torch.no_grad():
                teacher_logits = teacher_model(input_ids=inp, attention_mask=mask).logits

            student_out = student_model(input_ids=inp, attention_mask=mask, labels=lbl)
            student_logits = student_out.logits

            # Knowledge distillation loss = soft KL loss + hard Cross-Entropy loss
            loss_soft = kl_loss(
                F.log_softmax(student_logits / temp, dim=-1),
                F.softmax(teacher_logits / temp, dim=-1)
            ) * (temp ** 2)
            loss = alpha * loss_soft + (1 - alpha) * student_out.loss
            loss.backward()
            opt_student.step()

    # Move models to CPU for fair latency comparison and Dynamic INT8 Quantization
    teacher_model.cpu().eval()
    student_model.cpu().eval()

    student_dir = MODELS_DIR / "student_fp32"
    student_model.save_pretrained(student_dir)

    quantized_student = torch.ao.quantization.quantize_dynamic(
        student_model, {nn.Linear}, dtype=torch.qint8
    )
    quantized_path = MODELS_DIR / "quantized_student_int8.pt"
    torch.save(quantized_student, quantized_path)

    # =========================================================================
    # 5. Evaluate All 4 Models on the Same Held-Out Test Set (CPU)
    # =========================================================================
    print("Evaluating all 4 models on the held-out test set...")

    def pred_m1(text):
        return int(clf.predict(vectorizer.transform([text]))[0])

    def pred_m2(text):
        vec = torch.tensor(sentence_vector(text.split(), w2v), dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            return int(torch.argmax(ffnn(vec), dim=-1).item())

    def pred_m3(text):
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=48)
        with torch.no_grad():
            return int(torch.argmax(teacher_model(**enc).logits, dim=-1).item())

    def pred_m4(text):
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=48)
        with torch.no_grad():
            # DistilBERT has no token_type_ids -> pass only the inputs it accepts
            logits = quantized_student(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]).logits
            return int(torch.argmax(logits, dim=-1).item())

    def pred_m4_fp32(text):
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=48)
        with torch.no_grad():
            logits = student_model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]).logits
            return int(torch.argmax(logits, dim=-1).item())

    m1_res = measure_metrics_and_latency(pred_m1, X_test_text, y_test, tfidf_path)
    m2_res = measure_metrics_and_latency(pred_m2, X_test_text, y_test, ffnn_path)
    m3_res = measure_metrics_and_latency(pred_m3, X_test_text, y_test, teacher_dir)
    m4_res = measure_metrics_and_latency(pred_m4, X_test_text, y_test, quantized_path)
    # Extra row (not a required model): student before quantization, to isolate the effect of INT8
    m4f_res = measure_metrics_and_latency(pred_m4_fp32, X_test_text, y_test, student_dir)

    results = {
        "hardware": hardware_description(),
        "n_train": len(train_raw), "n_val": len(val_raw), "n_test": len(test_raw),
        "latency_runs": 100,
        "models": [
            {"model": "1. TF-IDF + Logistic Regression", "accuracy": m1_res[0], "oos_recall": m1_res[1], "f1": m1_res[2], "size_mb": m1_res[3], "latency_ms": m1_res[4]},
            {"model": "2. Word2Vec + SGD Feedforward NN", "accuracy": m2_res[0], "oos_recall": m2_res[1], "f1": m2_res[2], "size_mb": m2_res[3], "latency_ms": m2_res[4]},
            {"model": "3. Fine-Tuned BERT (Teacher FP32)", "accuracy": m3_res[0], "oos_recall": m3_res[1], "f1": m3_res[2], "size_mb": m3_res[3], "latency_ms": m3_res[4]},
            {"model": "4a. Distilled DistilBERT (FP32, before quantization) [extra]", "accuracy": m4f_res[0], "oos_recall": m4f_res[1], "f1": m4f_res[2], "size_mb": m4f_res[3], "latency_ms": m4f_res[4]},
            {"model": "4. Distilled + Quantized BERT (INT8)", "accuracy": m4_res[0], "oos_recall": m4_res[1], "f1": m4_res[2], "size_mb": m4_res[3], "latency_ms": m4_res[4]}
        ]
    }

    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n--- Final Results Table (R4) ---")
    for row in results["models"]:
        print(row)


if __name__ == "__main__":
    main()
