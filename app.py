"""
SWE015 Homework: Cargo Support Ticket Router (Use Case 2)
Student: Eren Acar (2309111059)

Run command:
    python app.py
"""

import pickle
import sys
import time
from pathlib import Path

import gradio as gr
import numpy as np
import torch
import torch.nn.functional as F
from gensim.models import Word2Vec
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from dataset import INTENT_NAMES, TEAM_MAP
from preprocessing import preprocess_message
from train_models import SimpleFFNN, sentence_vector

MODELS_DIR = Path("models")

REQUIRED = ["tfidf_baseline.pkl", "word2vec.model", "ffnn_sgd.pt", "bert_teacher", "quantized_student_int8.pt"]
missing = [n for n in REQUIRED if not (MODELS_DIR / n).exists()]
if missing:
    sys.exit(
        f"Missing trained model files in ./models: {missing}\n"
        "Either download the models folder (link in README.md) or run `python train_models.py`."
    )

# Load trained models
with open(MODELS_DIR / "tfidf_baseline.pkl", "rb") as f:
    tfidf_bundle = pickle.load(f)
vectorizer = tfidf_bundle["vectorizer"]
logreg_clf = tfidf_bundle["clf"]

w2v_model = Word2Vec.load(str(MODELS_DIR / "word2vec.model"))
ffnn_model = SimpleFFNN(input_dim=100, hidden_dim=64, num_classes=len(INTENT_NAMES))
ffnn_model.load_state_dict(torch.load(MODELS_DIR / "ffnn_sgd.pt", map_location="cpu"))
ffnn_model.eval()

tokenizer = AutoTokenizer.from_pretrained(MODELS_DIR / "bert_teacher")
teacher_bert = AutoModelForSequenceClassification.from_pretrained(MODELS_DIR / "bert_teacher")
teacher_bert.eval()

quantized_student = torch.load(MODELS_DIR / "quantized_student_int8.pt", map_location="cpu", weights_only=False)
quantized_student.eval()


def format_decision(probs, latency_ms, threshold):
    pred_id = int(np.argmax(probs))
    intent = INTENT_NAMES[pred_id]
    confidence = float(probs[pred_id])

    # Route low-confidence or out-of-scope messages to a human agent
    if intent == "out_of_scope" or confidence < threshold:
        routed_team = "Human Agent (Escalation)"
    else:
        routed_team = TEAM_MAP[intent]

    return intent, f"{confidence * 100:.1f}%", routed_team, f"{latency_ms:.2f} ms"


def route_support_ticket(message, threshold):
    if not message or not message.strip():
        return "Please enter a message.", "-", "-", "-", []

    prep = preprocess_message(message)
    clean_text = prep["clean_text"]
    tokens = prep["tokens"]

    codes_str = ", ".join(prep["extracted_codes"]) if prep["extracted_codes"] else "None"
    typos_str = ", ".join(prep["corrections"]) if prep["corrections"] else "None"

    # If the message is just a greeting/small talk, return the ELIZA rule response
    if prep["small_talk_reply"]:
        banner = f"Small Talk Detected (ELIZA Rule): {prep['small_talk_reply']}"
        return banner, clean_text, codes_str, typos_str, []

    rows = []

    # 1. Classical Baseline: TF-IDF + Logistic Regression
    t0 = time.perf_counter()
    p1 = logreg_clf.predict_proba(vectorizer.transform([clean_text]))[0]
    lat1 = (time.perf_counter() - t0) * 1000
    rows.append(["1. TF-IDF + Logistic Regression", *format_decision(p1, lat1, threshold)])

    # 2. Neural Network: Word2Vec + SGD Feedforward NN
    t0 = time.perf_counter()
    vec = torch.tensor(sentence_vector(tokens, w2v_model), dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        p2 = F.softmax(ffnn_model(vec), dim=-1)[0].numpy()
    lat2 = (time.perf_counter() - t0) * 1000
    rows.append(["2. Word2Vec + SGD Feedforward NN", *format_decision(p2, lat2, threshold)])

    # 3. Fine-Tuned BERT (Teacher FP32)
    t0 = time.perf_counter()
    enc = tokenizer(clean_text, return_tensors="pt", truncation=True, max_length=48)
    with torch.no_grad():
        p3 = F.softmax(teacher_bert(**enc).logits, dim=-1)[0].numpy()
    lat3 = (time.perf_counter() - t0) * 1000
    rows.append(["3. Fine-Tuned BERT (Teacher FP32)", *format_decision(p3, lat3, threshold)])

    # 4. Quantized Student Transformer (Dynamic INT8) - Primary Model (R5)
    t0 = time.perf_counter()
    enc_q = tokenizer(clean_text, return_tensors="pt", truncation=True, max_length=48)
    with torch.no_grad():
        # DistilBERT has no token_type_ids -> pass only the inputs it accepts
        logits_q = quantized_student(input_ids=enc_q["input_ids"], attention_mask=enc_q["attention_mask"]).logits
        p4 = F.softmax(logits_q, dim=-1)[0].numpy()
    lat4 = (time.perf_counter() - t0) * 1000
    q_intent, q_conf, q_team, q_lat = format_decision(p4, lat4, threshold)
    rows.append(["4. Distilled + Quantized BERT (INT8) [ACTIVE]", q_intent, q_conf, q_team, q_lat])

    primary_summary = f"Routed Team: {q_team}  |  Predicted Intent: {q_intent} ({q_conf})  |  Quantized Latency: {q_lat}"
    return primary_summary, clean_text, codes_str, typos_str, rows


with gr.Blocks(title="SWE015 - Cargo Support Ticket Router") as demo:
    gr.Markdown("## 📦 SWE015 Homework: Cargo Support Ticket Router (Use Case 2)")
    gr.Markdown("**Student:** Eren Acar (`2309111059`) — Compares 4 models side-by-side and routes tickets using the **Quantized Transformer (INT8)**.")

    with gr.Row():
        with gr.Column(scale=2):
            msg_input = gr.Textbox(
                label="Customer Support Message",
                value="My parcl TRK-84920 has been stuck at the sorting warehouse for 5 days, dlivery is very delayed!",
                lines=3
            )
            thresh_slider = gr.Slider(
                minimum=0.30, maximum=0.90, value=0.55, step=0.05,
                label="Confidence Threshold (Below threshold -> Human Agent)"
            )
            submit_btn = gr.Button("Analyze & Route Ticket", variant="primary")

        with gr.Column(scale=1):
            clean_out = gr.Textbox(label="Normalized Text (Area B)", interactive=False)
            codes_out = gr.Textbox(label="Extracted Tracking / Order Codes (Regex)", interactive=False)
            typos_out = gr.Textbox(label="Spelling Corrections (Minimum Edit Distance)", interactive=False)

    primary_out = gr.Textbox(label="Primary Production Decision (Model 4: Quantized INT8 Transformer)", interactive=False)

    comparison_table = gr.Dataframe(
        headers=["Model", "Predicted Intent", "Confidence", "Routed Team", "Latency"],
        label="Side-by-Side Comparison of All 4 Models",
        interactive=False
    )

    gr.Examples(
        examples=[
            ["My parcl TRK-84920 has been stuck at the sorting warehouse for 5 days, dlivery is very delayed!", 0.55],
            ["The coureir never rang my doorbell and left a missed delivery note for TRK-3019!", 0.55],
            ["My package ORD-5521 arrived completely crushed and the glass inside is broken.", 0.55],
            ["Please change the delivery address for shipment TRK-9912 to my office street.", 0.55],
            ["My dishwasher does not drain water, can you send a technician?", 0.55],
            ["hi, my parcel TRK-7788 is lost", 0.55],
            ["Hello, how are you today?", 0.55]
        ],
        inputs=[msg_input, thresh_slider]
    )

    submit_btn.click(
        fn=route_support_ticket,
        inputs=[msg_input, thresh_slider],
        outputs=[primary_out, clean_out, codes_out, typos_out, comparison_table]
    )

if __name__ == "__main__":
    demo.launch()
