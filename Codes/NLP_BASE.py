#!/usr/bin/env python3
# NLP_CASE.py - memory-optimized pipeline for <=4GB GPUs

import os
import re
import argparse
from pathlib import Path
from typing import List

import pandas as pd
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForMaskedLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


# -------------------------
# Text cleaning utility
# -------------------------
def clean_text(s: str) -> str:
    if s is None:
        return ""
    s = re.sub(r"\s+", " ", s.replace("\n", " ").replace("\r", " ")).strip()
    s = re.sub(r"http\S+", "", s)
    s = re.sub(r"[^A-Za-z0-9\s\.\,\-\$%€£:;()\/]", " ", s)
    s = re.sub(r"\s{2,}", " ", s)
    return s.strip()


# -------------------------
# Reuters loader
# -------------------------
def load_reuters_with_labels(base_path: str) -> pd.DataFrame:
    base = Path(base_path)
    if not base.exists():
        print(f"[WARNING] Reuters path missing: {base_path}")
        return pd.DataFrame()
    rows = []
    for fp in base.rglob("*.txt"):
        try:
            txt = fp.read_text(encoding="utf-8", errors="ignore").strip()
            if not txt:
                continue
            label = fp.parent.name
            rows.append({"text": txt, "label": label})
        except Exception:
            continue
    return pd.DataFrame(rows)


# -------------------------
# Financial PhraseBank loader
# -------------------------
def load_fpb_texts(base_path: str) -> pd.DataFrame:
    base = Path(base_path)
    if not base.exists():
        print(f"[WARNING] FPB path missing: {base_path}")
        return pd.DataFrame()
    texts = []
    for fp in base.rglob("*.txt"):
        try:
            raw = fp.read_text(encoding="latin-1", errors="ignore")
            for line in raw.splitlines():
                s = line.strip()
                if s:
                    texts.append(s)
        except Exception:
            continue
    return pd.DataFrame({"text": texts})


# -------------------------
# FiQA loader (robust)
# -------------------------
def load_fiqa_from_hf(hf_path: str) -> pd.DataFrame:
    df = None
    try:
        df = pd.read_json(hf_path)
    except Exception:
        try:
            from datasets import load_dataset
            ds = load_dataset("llamafactory/fiqa")
            df = pd.DataFrame(ds["train"])
        except Exception as e:
            print(f"[WARNING] FiQA load failed: {e}")
            return pd.DataFrame()

    # Ensure we have a 'text' column
    if "text" in df.columns:
        return df[["text"]].copy()
    if all(c in df.columns for c in ["instruction", "input", "output"]):
        df["instruction"] = df["instruction"].fillna("")
        df["input"] = df["input"].fillna("")
        df["output"] = df["output"].fillna("")
        df["text"] = (
            df["instruction"].astype(str)
            + " "
            + df["input"].astype(str)
            + " "
            + df["output"].astype(str)
        ).str.strip()
        return df[["text"]].copy()
    return pd.DataFrame()


# -------------------------
# Financial Tweets loader
# -------------------------
def load_financial_tweets_kaggle() -> pd.DataFrame:
    try:
        import kagglehub
        from kagglehub import KaggleDatasetAdapter
        res = kagglehub.load_dataset(
            KaggleDatasetAdapter.PANDAS, "davidwallach/financial-tweets", None
        )
        if isinstance(res, pd.DataFrame):
            return res
        return pd.read_csv(str(res))
    except Exception as e:
        print(f"[WARNING] Financial tweets load failed: {e}")
        return pd.DataFrame(columns=["text", "label"])


# -------------------------
# Dataset classes
# -------------------------
class SupervisedTextDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_len=64):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        txt = str(self.texts[idx])
        enc = self.tokenizer(
            txt,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


# -------------------------
# Model
# -------------------------
class WSDEncoderDecoder(nn.Module):
    def __init__(self, encoder_name="prajjwal1/bert-tiny", hidden_size=128, num_labels=2, nhead=4, dec_layers=1, dropout=0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_name)
        try:
            self.encoder.gradient_checkpointing_enable()
        except Exception:
            pass

        enc_hidden = getattr(self.encoder.config, "hidden_size", 128)
        self.wsd_proj = nn.Sequential(
            nn.Linear(enc_hidden, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
        )
        self.hidden = hidden_size
        self.query = nn.Parameter(torch.randn(1, hidden_size))
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_size, nhead=nhead, dropout=dropout
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=dec_layers)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        enc = self.encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        memory = self.wsd_proj(enc.last_hidden_state)
        B = input_ids.size(0)
        tgt = self.query.unsqueeze(1).repeat(1, B, 1)
        mem = memory.permute(1, 0, 2)
        dec_out = self.decoder(tgt=tgt, memory=mem)
        dec_out = dec_out.squeeze(0)
        return self.classifier(dec_out)

    def replace_classifier(self, num_labels):
        self.classifier = nn.Linear(self.hidden, num_labels)


# -------------------------
# Evaluation
# -------------------------
def evaluate(model, dataloader, device):
    model.eval()
    preds, gold = [], []
    with torch.no_grad():
        for batch in dataloader:
            logits = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
            p = torch.argmax(logits, dim=-1).cpu().numpy().tolist()
            preds.extend(p)
            gold.extend(batch["labels"].cpu().numpy().tolist())
    acc = accuracy_score(gold, preds) if len(gold) else 0.0
    f1 = f1_score(gold, preds, average="macro") if len(gold) else 0.0
    return acc, f1


# -------------------------
# Training loop
# -------------------------
def train_supervised(model, train_loader, val_loader, device, epochs=1, lr=2e-5, out_dir="ckpt"):
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)
    best_f1 = -1.0
    os.makedirs(out_dir, exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))
        for batch in tqdm(train_loader, desc=f"Train epoch {epoch}", leave=False):
            optimizer.zero_grad()
            with torch.amp.autocast(device_type="cuda", enabled=(device.type == "cuda")):
                logits = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
                loss = F.cross_entropy(logits, batch["labels"].to(device))
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            total_loss += loss.item()
        avg_loss = total_loss / max(1, len(train_loader))
        acc, f1 = evaluate(model, val_loader, device)
        print(f"[Epoch {epoch}] loss={avg_loss:.4f} val_acc={acc} val_f1={f1}")
        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), os.path.join(out_dir, "best.pth"))
    torch.cuda.empty_cache()
    return os.path.join(out_dir, "best.pth")


# -------------------------
# Main pipeline
# -------------------------
def main(reuters_path, fpb_path, fiqa_hfpath, do_mlm_on_fpb, batch_size, max_len, reuters_epochs, fpb_mlm_epochs, fiqa_epochs, tweets_epochs, lr, out_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(out_dir, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")

    # Reuters
    reuters_df = load_reuters_with_labels(reuters_path)
    if reuters_df.empty:
        print("[WARNING] Reuters data missing. Skipping Reuters training.")
        return

    reuters_df["text"] = reuters_df["text"].astype(str).map(clean_text)
    le = LabelEncoder()
    reuters_df["label_id"] = le.fit_transform(reuters_df["label"].astype(str))
    train_df, val_df = train_test_split(reuters_df, test_size=0.2, stratify=reuters_df["label_id"])
    train_ds = SupervisedTextDataset(train_df["text"].tolist(), train_df["label_id"].tolist(), tokenizer, max_len)
    val_ds = SupervisedTextDataset(val_df["text"].tolist(), val_df["label_id"].tolist(), tokenizer, max_len)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    model = WSDEncoderDecoder(encoder_name="prajjwal1/bert-tiny", hidden_size=128, num_labels=len(le.classes_), nhead=4)
    train_supervised(model, train_loader, val_loader, device, epochs=reuters_epochs, lr=lr, out_dir=os.path.join(out_dir, "reuters"))

    # FiQA
    fiqa_df = load_fiqa_from_hf(fiqa_hfpath)
    if not fiqa_df.empty:
        fiqa_df["text"] = fiqa_df["text"].astype(str).map(clean_text)
        fiqa_df["label_id"] = 0
        train_df, val_df = train_test_split(fiqa_df, test_size=0.1)
        train_ds = SupervisedTextDataset(train_df["text"].tolist(), train_df["label_id"].tolist(), tokenizer, max_len)
        val_ds = SupervisedTextDataset(val_df["text"].tolist(), val_df["label_id"].tolist(), tokenizer, max_len)
        train_loader = DataLoader(train_ds, batch_size=batch_size)
        val_loader = DataLoader(val_ds, batch_size=batch_size)
        model.replace_classifier(num_labels=1)
        model = model.to(device)
        train_supervised(model, train_loader, val_loader, device, epochs=fiqa_epochs, lr=lr, out_dir=os.path.join(out_dir, "fiqa"))


# -------------------------
# CLI entry
# -------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reuters_path", type=str, default="/home/sanjith-ganesa/Desktop/SEM_7/NLP/NLP_proj_data/reuters+transcribed+subset/ReutersTranscribedSubset")
    parser.add_argument("--fpb_path", type=str, default="/home/sanjith-ganesa/Desktop/SEM_7/NLP/NLP_proj_data/FinancialPhraseBank-v1.0/FinancialPhraseBank-v1.0")
    parser.add_argument("--fiqa_hfpath", type=str, default="hf://datasets/llamafactory/fiqa/train.json")
    parser.add_argument("--out_dir", type=str, default="./wsd_pipeline_out_tiny")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_len", type=int, default=64)
    parser.add_argument("--reuters_epochs", type=int, default=1)
    parser.add_argument("--fpb_mlm_epochs", type=int, default=1)
    parser.add_argument("--fiqa_epochs", type=int, default=1)
    parser.add_argument("--tweets_epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--do_mlm_on_fpb", action="store_true")
    args = parser.parse_args()

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:32")

    main(
        reuters_path=args.reuters_path,
        fpb_path=args.fpb_path,
        fiqa_hfpath=args.fiqa_hfpath,
        do_mlm_on_fpb=args.do_mlm_on_fpb,
        batch_size=args.batch_size,
        max_len=args.max_len,
        reuters_epochs=args.reuters_epochs,
        fpb_mlm_epochs=args.fpb_mlm_epochs,
        fiqa_epochs=args.fiqa_epochs,
        tweets_epochs=args.tweets_epochs,
        lr=args.lr,
        out_dir=args.out_dir,
    )
