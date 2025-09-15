#!/usr/bin/env python3
# NLP_CASE.py - CPU-only pipeline with evaluation metrics:
# Directional Agreement (DA), Event-Impact Correlation (EIC),
# Financial Sense Consistency (FSC), Profitability-Oriented Measure (Backtest)

import os
import re
import argparse
from pathlib import Path
from typing import List, Optional, Tuple, Dict

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
from scipy.stats import spearmanr
from sklearn.metrics.pairwise import cosine_similarity

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
            # not all models have this method; ignore if not available
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

    def forward(self, input_ids, attention_mask, return_embedding: bool = False):
        enc = self.encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        memory = self.wsd_proj(enc.last_hidden_state)  # [B, T, H]
        B = input_ids.size(0)
        tgt = self.query.unsqueeze(1).repeat(1, B, 1)  # [1, B, H]
        mem = memory.permute(1, 0, 2)  # [T, B, H]
        dec_out = self.decoder(tgt=tgt, memory=mem)  # [1, B, H]
        dec_out = dec_out.squeeze(0)  # [B, H]
        logits = self.classifier(dec_out)  # [B, C]
        if return_embedding:
            return logits, dec_out.detach().cpu().numpy()
        return logits

    def replace_classifier(self, num_labels):
        self.classifier = nn.Linear(self.hidden, num_labels)


# -------------------------
# Metrics: DA, EIC, FSC, Backtest
# -------------------------
def directional_of_labels(gold: List[int], label_map: Optional[Dict[int, int]] = None) -> np.ndarray:
    """
    Map discrete labels to directional values -1,0,1.
    If label_map provided, uses it; otherwise heuristics:
      - if 2 classes: map 0 -> -1, 1 -> +1
      - if 3 classes: map [0,1,2] -> [-1,0,+1] (assumes ordered)
      - else: tries to map first class -> -1, last -> +1, others -> 0
    """
    uniq = sorted(list(set(gold)))
    if label_map is None:
        label_map = {}
        if len(uniq) == 2:
            label_map[uniq[0]] = -1
            label_map[uniq[1]] = 1
        elif len(uniq) == 3:
            label_map[uniq[0]] = -1
            label_map[uniq[1]] = 0
            label_map[uniq[2]] = 1
        else:
            # assign first negative, last positive, others neutral
            for i, v in enumerate(uniq):
                if i == 0:
                    label_map[v] = -1
                elif i == len(uniq) - 1:
                    label_map[v] = 1
                else:
                    label_map[v] = 0
    return np.array([label_map.get(x, 0) for x in gold]), label_map


def compute_directional_agreement(gold: List[int], preds: List[int]) -> float:
    """
    DA = proportion where predicted direction == gold direction
    Uses heuristics to map labels to directions.
    """
    if len(gold) == 0:
        return 0.0
    gold_dirs, label_map = directional_of_labels(gold)
    pred_dirs = np.array([label_map.get(p, 0) for p in preds])
    # agreement when directions equal
    agree = (gold_dirs == pred_dirs).sum()
    return float(agree) / len(gold)


def compute_event_impact_correlation(preds: List[int], logits: np.ndarray, impact_scores: Optional[List[float]], gold: Optional[List[int]] = None) -> Optional[float]:
    """
    EIC: Spearman correlation between model's predicted signal strength and provided impact scores.
    Signal strength computed as (prob_pos - prob_neg). Heuristics:
      - If binary classification, use prob of positive class.
      - If multiclass, compute (prob_max - prob_min) or prob of positive class if identifiable.
    Returns Spearman rho if impact_scores provided, else None.
    """
    if impact_scores is None:
        print("[EIC] impact_scores not provided; skipping EIC.")
        return None
    if logits is None or len(logits) == 0:
        print("[EIC] no logits available; skipping EIC.")
        return None
    probs = softmax_rows(logits)
    # try identify pos/neg using gold if present; else assume last class is positive
    num_classes = probs.shape[1]
    if num_classes == 1:
        signals = probs[:, 0]
    elif num_classes == 2:
        # signal = prob(class1) - prob(class0) (assume class order)
        signals = probs[:, 1] - probs[:, 0]
    else:
        # signal: prob(last class) - prob(first class)
        signals = probs[:, -1] - probs[:, 0]
    # compute spearman
    try:
        rho, p = spearmanr(signals, np.array(impact_scores[: len(signals)]))
        return float(rho)
    except Exception as e:
        print("[EIC] spearmanr failed:", e)
        return None


def softmax_rows(logits: np.ndarray) -> np.ndarray:
    x = np.array(logits, dtype=np.float64)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / e_x.sum(axis=1, keepdims=True)


def compute_fsc(embeddings: np.ndarray, preds: List[int]) -> Optional[float]:
    """
    FSC: Financial Sense Consistency.
    Heuristic: for each predicted class, compute centroid of embeddings for that class,
    then compute average cosine similarity between each sample's embedding and its class centroid.
    Return average similarity across all samples (range -1..1).
    """
    if embeddings is None or len(embeddings) == 0:
        print("[FSC] embeddings not available; skipping FSC.")
        return None
    emb = np.array(embeddings)
    preds = np.array(preds)
    unique = np.unique(preds)
    sims = []
    for cls in unique:
        idx = np.where(preds == cls)[0]
        if len(idx) == 0:
            continue
        cls_emb = emb[idx]
        centroid = cls_emb.mean(axis=0, keepdims=True)  # [1, d]
        # cosine similarity between centroid and each sample
        cs = cosine_similarity(cls_emb, centroid).flatten()
        sims.extend(cs.tolist())
    if len(sims) == 0:
        return None
    return float(np.mean(sims))


def compute_backtest(preds: List[int], logits: np.ndarray, future_returns: Optional[List[float]], gold: Optional[List[int]] = None) -> Optional[Dict[str, float]]:
    """
    Simple backtest:
    - Map preds to signals (-1,0,+1) using the same heuristic as directional_of_labels
    - If future_returns provided: compute strategy returns = signal * future_returns
    - Report cumulative return and Sharpe-like ratio (mean/std * sqrt(252) if daily)
    Returns dict with cumulative_return, mean_return, std_return, sharpe
    """
    if future_returns is None:
        print("[BACKTEST] future_returns not provided; skipping backtest.")
        return None
    if logits is None:
        # still can use preds but no probability weighting
        pass
    gold_dirs, label_map = directional_of_labels(gold if gold is not None else preds)
    # compute pred directions using same label_map
    pred_dirs = np.array([label_map.get(p, 0) for p in preds])
    returns = np.array(future_returns[: len(pred_dirs)], dtype=np.float64)
    strat_returns = pred_dirs * returns
    if strat_returns.size == 0:
        return None
    cum_return = float(np.nansum(strat_returns))
    mean_r = float(np.nanmean(strat_returns))
    std_r = float(np.nanstd(strat_returns, ddof=1)) if strat_returns.size > 1 else 0.0
    sharpe = float((mean_r / std_r) * (252 ** 0.5)) if std_r > 0 else float("nan")
    return {"cumulative_return": cum_return, "mean_return": mean_r, "std_return": std_r, "sharpe": sharpe}


# -------------------------
# Evaluation that returns embeddings and logits
# -------------------------
def evaluate_with_embeddings(model, dataloader, device) -> Tuple[List[int], List[int], np.ndarray, np.ndarray]:
    model.eval()
    preds = []
    gold = []
    logits_all = []
    embeddings_all = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].cpu().numpy().tolist()
            out = model(input_ids, attention_mask, return_embedding=True)
            if isinstance(out, tuple):
                logits_batch, emb_batch = out
            else:
                logits_batch = out
                emb_batch = None
            if isinstance(logits_batch, torch.Tensor):
                logits_np = logits_batch.cpu().numpy()
            else:
                logits_np = np.array(logits_batch)
            preds_batch = np.argmax(logits_np, axis=-1).tolist()
            preds.extend(preds_batch)
            gold.extend(labels)
            logits_all.append(logits_np)
            if emb_batch is not None:
                embeddings_all.append(emb_batch)
    if len(logits_all):
        logits_all = np.vstack(logits_all)
    else:
        logits_all = np.zeros((0, 0))
    if len(embeddings_all):
        embeddings_all = np.vstack(embeddings_all)
    else:
        embeddings_all = np.zeros((0, 0))
    return preds, gold, logits_all, embeddings_all


# -------------------------
# Training loop (CPU-only)
# -------------------------
def train_supervised(model, train_loader, val_loader, device, epochs=1, lr=2e-5, out_dir="ckpt"):
    # CPU training: do not use mixed-precision or CUDA
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)
    best_f1 = -1.0
    os.makedirs(out_dir, exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Train epoch {epoch}", leave=False):
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            logits = model(input_ids, attention_mask)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += float(loss.item())
        avg_loss = total_loss / max(1, len(train_loader))
        # evaluate and compute classic metrics
        preds, gold, logits_all, embeddings_all = evaluate_with_embeddings(model, val_loader, device)
        acc = accuracy_score(gold, preds) if len(gold) else 0.0
        f1 = f1_score(gold, preds, average="macro") if len(gold) else 0.0
        print(f"[Epoch {epoch}] loss={avg_loss:.4f} val_acc={acc:.4f} val_f1={f1:.4f}")
        # compute advanced metrics if possible
        # try to extract auxiliary columns from dataloader.dataset (if present)
        aux_impact = None
        aux_returns = None
        try:
            ds = val_loader.dataset
            if hasattr(ds, "dataframe"):
                df = ds.dataframe
                if "impact_score" in df.columns:
                    aux_impact = df["impact_score"].tolist()
                if "future_return" in df.columns:
                    aux_returns = df["future_return"].tolist()
        except Exception:
            pass
        da = compute_directional_agreement(gold, preds)
        eic = compute_event_impact_correlation(preds, logits_all, aux_impact, gold)
        fsc = compute_fsc(embeddings_all, preds)
        backtest = compute_backtest(preds, logits_all, aux_returns, gold)
        print(f"  DA={da:.4f}  EIC={ (f'{eic:.4f}' if eic is not None else 'N/A') }  FSC={ (f'{fsc:.4f}' if fsc is not None else 'N/A') }")
        if backtest is not None:
            print(f"  BACKTEST cum_ret={backtest['cumulative_return']:.4f} mean={backtest['mean_return']:.6f} std={backtest['std_return']:.6f} sharpe={backtest['sharpe']:.4f}")
        # save best
        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), os.path.join(out_dir, "best.pth"))
            print("Saved best ->", os.path.join(out_dir, "best.pth"))
    return os.path.join(out_dir, "best.pth")


# -------------------------
# Main pipeline
# -------------------------
def main(
    reuters_path: str,
    fpb_path: str,
    fiqa_hfpath: str,
    do_mlm_on_fpb: bool,
    batch_size: int,
    max_len: int,
    reuters_epochs: int,
    fpb_mlm_epochs: int,
    fiqa_epochs: int,
    tweets_epochs: int,
    lr: float,
    out_dir: str,
):
    # force CPU-only
    device = torch.device("cpu")
    print("Device forced to CPU:", device)
    os.makedirs(out_dir, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")

    # ---------------------------
    # Reuters (supervised)
    # ---------------------------
    reuters_df = load_reuters_with_labels(reuters_path)
    if reuters_df.empty:
        print("[WARNING] Reuters dataset is empty or path wrong. Exiting.")
        return
    reuters_df["text"] = reuters_df["text"].astype(str).map(clean_text)
    le = LabelEncoder()
    reuters_df["label_id"] = le.fit_transform(reuters_df["label"].astype(str))
    train_df, val_df = train_test_split(reuters_df, test_size=0.2, stratify=reuters_df["label_id"], random_state=42)
    # attach dataframe to dataset for auxiliary columns detection in training loop
    train_ds = SupervisedTextDataset(train_df["text"].tolist(), train_df["label_id"].tolist(), tokenizer, max_len)
    val_ds = SupervisedTextDataset(val_df["text"].tolist(), val_df["label_id"].tolist(), tokenizer, max_len)
    # attach for aux detection (optional)
    setattr(train_ds, "dataframe", train_df.reset_index(drop=True))
    setattr(val_ds, "dataframe", val_df.reset_index(drop=True))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = WSDEncoderDecoder(encoder_name="prajjwal1/bert-tiny", hidden_size=128, num_labels=len(le.classes_), nhead=4)
    print("Training on Reuters (CPU-only)...")
    train_supervised(model, train_loader, val_loader, device, epochs=reuters_epochs, lr=lr, out_dir=os.path.join(out_dir, "reuters"))

    # ---------------------------
    # (Optional) MLM on FPB - using HF Trainer (still CPU)
    # ---------------------------
    if do_mlm_on_fpb:
        fpb_df = load_fpb_texts(fpb_path)
        if not fpb_df.empty:
            print("FPB loaded, but MLM Trainer may require 'accelerate' and can be slow on CPU.")
            # We will attempt a very small MLM run if requested; otherwise skip
            try:
                from datasets import Dataset as HFDataset
                hf_list = [{"text": t} for t in fpb_df["text"].tolist()[:2000]]  # small subset for CPU
                hf_ds = HFDataset.from_list(hf_list)
                def tok(ex):
                    return tokenizer(ex["text"], truncation=True, padding="max_length", max_length=max_len)
                tokenized = hf_ds.map(tok, batched=True, remove_columns=["text"])
                tokenized.set_format(type="torch", columns=["input_ids", "attention_mask"])
                mlm_model = AutoModelForMaskedLM.from_pretrained("prajjwal1/bert-tiny")
                data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)
                training_args = TrainingArguments(output_dir=os.path.join(out_dir, "fpb_mlm"), num_train_epochs=1, per_device_train_batch_size=1, logging_steps=50, save_strategy="epoch")
                trainer = Trainer(model=mlm_model, args=training_args, train_dataset=tokenized, data_collator=data_collator)
                print("Starting a tiny MLM run on FPB (CPU). This will be slow; skipping if Trainer fails.")
                trainer.train()
                trainer.save_model(os.path.join(out_dir, "fpb_mlm"))
            except Exception as e:
                print("FPB MLM step skipped or failed (likely due to CPU/time/accelerate):", e)
        else:
            print("FPB empty or missing; skipping MLM.")
    else:
        print("Skipping FPB MLM (not requested).")

    # ---------------------------
    # FiQA fine-tune (optional)
    # ---------------------------
    fiqa_df = load_fiqa_from_hf(fiqa_hfpath)
    if fiqa_df is None or fiqa_df.empty:
        print("FiQA dataset empty or not loaded; skipping FiQA fine-tuning.")
    else:
        fiqa_df["text"] = fiqa_df["text"].astype(str).map(clean_text)
        # FiQA may be unlabeled; put dummy label 0
        fiqa_df["label_id"] = 0
        # if user had labels, the loader above would have preserved them; we assume none
        train_df, val_df = train_test_split(fiqa_df, test_size=0.1, random_state=42)
        train_ds = SupervisedTextDataset(train_df["text"].tolist(), train_df["label_id"].tolist(), tokenizer, max_len)
        val_ds = SupervisedTextDataset(val_df["text"].tolist(), val_df["label_id"].tolist(), tokenizer, max_len)
        setattr(train_ds, "dataframe", train_df.reset_index(drop=True))
        setattr(val_ds, "dataframe", val_df.reset_index(drop=True))
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
        # adapt classifier to number of labels (1 in this case)
        model.replace_classifier(num_labels=max(1, int(fiqa_df["label_id"].nunique())))
        model = model.to(device)
        print("Fine-tuning on FiQA (CPU-only)...")
        train_supervised(model, train_loader, val_loader, device, epochs=fiqa_epochs, lr=lr, out_dir=os.path.join(out_dir, "fiqa"))

    # ---------------------------
    # Financial Tweets (optional)
    # ---------------------------
    tweets_df = load_financial_tweets_kaggle()
    if tweets_df is None or tweets_df.empty:
        print("Financial tweets not loaded or empty; skipping tweets fine-tuning.")
    else:
        # try to standardize column names
        if "text" not in tweets_df.columns:
            possible = [c for c in tweets_df.columns if "text" in c.lower()]
            if possible:
                tweets_df = tweets_df.rename(columns={possible[0]: "text"})
        if "label" not in tweets_df.columns:
            possible = [c for c in tweets_df.columns if "sent" in c.lower() or "label" in c.lower()]
            if possible:
                tweets_df = tweets_df.rename(columns={possible[0]: "label"})
        tweets_df.dropna(subset=["text"], inplace=True)
        tweets_df["text"] = tweets_df["text"].astype(str).map(clean_text)
        if "label" in tweets_df.columns:
            le_tweets = LabelEncoder()
            tweets_df["label_id"] = le_tweets.fit_transform(tweets_df["label"].astype(str))
        else:
            tweets_df["label_id"] = 0
        train_df, val_df = train_test_split(tweets_df, test_size=0.1, random_state=42)
        train_ds = SupervisedTextDataset(train_df["text"].tolist(), train_df["label_id"].tolist(), tokenizer, max_len)
        val_ds = SupervisedTextDataset(val_df["text"].tolist(), val_df["label_id"].tolist(), tokenizer, max_len)
        setattr(train_ds, "dataframe", train_df.reset_index(drop=True))
        setattr(val_ds, "dataframe", val_df.reset_index(drop=True))
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
        model.replace_classifier(num_labels=max(1, int(tweets_df["label_id"].nunique())))
        model = model.to(device)
        print("Fine-tuning on Financial Tweets (CPU-only)...")
        train_supervised(model, train_loader, val_loader, device, epochs=tweets_epochs, lr=lr, out_dir=os.path.join(out_dir, "tweets"))

    print("Pipeline finished. Checkpoints (if any) are in:", out_dir)


# -------------------------
# CLI entry
# -------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reuters_path", type=str, default="/home/sanjith-ganesa/Desktop/SEM_7/NLP/NLP_proj_data/reuters+transcribed+subset/ReutersTranscribedSubset")
    parser.add_argument("--fpb_path", type=str, default="/home/sanjith-ganesa/Desktop/SEM_7/NLP/NLP_proj_data/FinancialPhraseBank-v1.0/FinancialPhraseBank-v1.0")
    parser.add_argument("--fiqa_hfpath", type=str, default="hf://datasets/llamafactory/fiqa/train.json")
    parser.add_argument("--out_dir", type=str, default="./wsd_pipeline_out_tiny_cpu")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_len", type=int, default=64)
    parser.add_argument("--reuters_epochs", type=int, default=1)
    parser.add_argument("--fpb_mlm_epochs", type=int, default=1)
    parser.add_argument("--fiqa_epochs", type=int, default=1)
    parser.add_argument("--tweets_epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--do_mlm_on_fpb", action="store_true")
    args = parser.parse_args()

    # Ensure CPU-only run (user requested)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:32")

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
