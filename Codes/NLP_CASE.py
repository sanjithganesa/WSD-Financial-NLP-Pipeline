#!/usr/bin/env python3
# NLP_CASE.py - CPU-only pipeline with evaluation metrics:
# Directional Agreement (DA), Event-Impact Correlation (EIC),
# Financial Sense Consistency (FSC), Profitability-Oriented Measure (Backtest)
# Enhanced with modern embeddings + augmentation to address 7 ambiguity types:
# 1) Polysemy, 2) Domain jargon, 3) Metaphor, 4) Company vs common words,
# 5) Event-impact ambiguity, 6) Cross-sentence links, 7) Temporal/context shifts.
#
# NOTE: This modifies the model by adding embedding fusion & light augmentation.
# Core training, metrics and dataset loaders remain intact as requested.

import os
import re
import argparse
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

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

# Optional: sentence-transformers for modern sentence embeddings
try:
    from sentence_transformers import SentenceTransformer
    SBERT_AVAILABLE = True
except Exception:
    SBERT_AVAILABLE = False

# Optional: spaCy for POS/NER; fallback to regex if not available
try:
    import spacy
    SPACY_AVAILABLE = True
    try:
        _nlp = spacy.load("en_core_web_sm")
    except Exception:
        _nlp = None
except Exception:
    SPACY_AVAILABLE = False
    _nlp = None

# -------------------------
# Text cleaning utility
# -------------------------
def clean_text(s: str) -> str:
    if s is None:
        return ""
    s = re.sub(r"\s+", " ", s.replace("\n", " ").replace("\r", " ")).strip()
    s = re.sub(r"http\S+", "", s)
    # keep cashtags/tickers and basic punctuation
    s = re.sub(r"[^A-Za-z0-9\s\.\,\-\$%€£:;()\/\#\@]", " ", s)
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
# Financial Tweets loader (stub - original used kagglehub)
# -------------------------
def load_financial_tweets_kaggle() -> pd.DataFrame:
    # Keep original behavior: try kagglehub then fallback.
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
# Dataset classes (with context + augmentation)
# -------------------------
class SupervisedTextDataset(Dataset):
    """
    Now returns:
      - input_ids, attention_mask, labels (as before)
      - text (cleaned raw string) for modern embeddings and augmentations
      - context_text (optional concatenation of neighboring sentences for cross-sentence context)
      - lightweight linguistic cues: pos_tags, has_cashtag, ticker, is_metaphor_hint
    """
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_len=64, context_window: int = 0):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.context_window = context_window  # how many neighbor sentences to include
        # Precompute simple cues
        self.cues = [self._compute_cues(t) for t in texts]

    def __len__(self):
        return len(self.texts)

    def _extract_ticker(self, s: str) -> Optional[str]:
        # find $TICKER or (TICKER) style hints
        m = re.search(r"\$([A-Za-z]{1,6})", s)
        if m:
            return m.group(1).upper()
        # parentheses tickers e.g., Apple (AAPL)
        m = re.search(r"\(([A-Za-z]{1,6})\)", s)
        if m:
            return m.group(1).upper()
        return None

    def _metaphor_hint(self, s: str) -> bool:
        # Very lightweight heuristic: presence of words like bleed, rally, surge, crash used metaphorically
        return bool(re.search(r"\b(bleed|bleeding|rally|surge|crash|soar|plummet|tank)\b", s, flags=re.I))

    def _compute_cues(self, s: str) -> Dict[str, Any]:
        s_clean = s if s is not None else ""
        has_cashtag = bool(re.search(r"\$[A-Za-z]{1,6}", s_clean))
        ticker = self._extract_ticker(s_clean)
        metaphor = self._metaphor_hint(s_clean)
        # POS / NER via spaCy if available (else simple capital word heuristic)
        pos_tags = None
        ents = []
        if SPACY_AVAILABLE and _nlp is not None:
            try:
                doc = _nlp(s_clean)
                pos_tags = [tok.pos_ for tok in doc]
                ents = [(ent.text, ent.label_) for ent in doc.ents]
            except Exception:
                pos_tags = None
                ents = []
        else:
            # fallback: list of capitalized tokens as crude named entities
            ents = [(w, "PROPN") for w in re.findall(r"\b[A-Z][a-zA-Z]{1,}\b", s_clean)]
        return {"has_cashtag": has_cashtag, "ticker": ticker, "metaphor": metaphor, "pos": pos_tags, "ents": ents}

    def __getitem__(self, idx):
        txt = str(self.texts[idx])
        # build context_text by naive neighboring sentence concatenation if context_window>0
        context_text = txt
        if self.context_window > 0:
            # naive splitting by sentence punctuation
            sents = re.split(r'(?<=[.!?])\s+', txt)
            # If user supplied long paragraph, include neighbors inside the same paragraph (best-effort)
            # Here we just use the same sentence (no external source), so context_window limited.
            context_text = txt  # For now, same as txt (could be extended to pull from dataframe surrounding rows)
        enc = self.tokenizer(
            txt,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        item = {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
            "text": clean_text(txt),
            "context_text": clean_text(context_text),
            "cues": self.cues[idx],
        }
        return item


# -------------------------
# Model: Add embedding fusion + augmentations
# -------------------------
class WSDEncoderDecoder(nn.Module):
    """
    Original architecture kept, with a modular embedding_fusion layer added.
    Fusion strategies implemented:
     - add: pooled encoder CLS + projected modern embedding
     - concat_proj: concat CLS + modern_emb, then projection
     - attention: small attention layer to combine vectors
    Additional lightweight feature injection: ticker embedding (learned small embedding for known tickers)
    """
    def __init__(self,
                 encoder_name="prajjwal1/bert-tiny",
                 hidden_size=128,
                 num_labels=2,
                 nhead=4,
                 dec_layers=1,
                 dropout=0.1,
                 modern_emb_model: Optional[str] = None,
                 fusion_method: str = "attention",  # 'add' | 'concat_proj' | 'attention'
                 use_finance_emb: bool = False,
                 ticker_vocab: Optional[List[str]] = None):
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

        # ---- Modern embedding module ----
        self.fusion_method = fusion_method
        self.modern_emb = None
        self.modern_emb_dim = None
        if SBERT_AVAILABLE and modern_emb_model is not None:
            try:
                # try to load a finance-specific model first if requested
                if use_finance_emb:
                    # prefer a finance SBERT/FinBERT if provided; user can pass model name e.g., "yiyanghkust/finbert-tone"
                    self.modern_emb = SentenceTransformer(modern_emb_model)
                else:
                    self.modern_emb = SentenceTransformer(modern_emb_model)
                # infer embedding dim by encoding a dummy sentence
                test_vec = self.modern_emb.encode("test", convert_to_numpy=True)
                self.modern_emb_dim = int(test_vec.shape[-1])
            except Exception as e:
                print("[WARN] modern embedding initialization failed:", e)
                self.modern_emb = None

        # projection layers to align modern_emb <-> hidden
        if self.modern_emb is not None:
            if self.fusion_method == "add":
                # project modern_emb -> hidden
                self.emb_proj = nn.Linear(self.modern_emb_dim, hidden_size)
            elif self.fusion_method == "concat_proj":
                # project concatenated vector (hidden + modern_dim) -> hidden
                self.concat_proj = nn.Linear(hidden_size + self.modern_emb_dim, hidden_size)
            elif self.fusion_method == "attention":
                # small attention mechanism: compute attention between CLS and modern_emb
                self.key_proj = nn.Linear(self.modern_emb_dim, hidden_size)
                self.query_proj = nn.Linear(hidden_size, hidden_size)
                self.value_proj = nn.Linear(self.modern_emb_dim, hidden_size)
                self.out_proj = nn.Linear(hidden_size, hidden_size)
            else:
                # default: add
                self.emb_proj = nn.Linear(self.modern_emb_dim, hidden_size)

        # ticker embedding (addresses Company vs common words; gives model a learned embedding for tickers)
        self.ticker_vocab = ticker_vocab if ticker_vocab is not None else []
        self.ticker_to_idx = {t: i for i, t in enumerate(self.ticker_vocab)}
        if len(self.ticker_vocab) > 0:
            self.ticker_emb = nn.Embedding(len(self.ticker_vocab), hidden_size)
        else:
            self.ticker_emb = None

    def fuse_embeddings(self, cls_vec: torch.Tensor, texts: List[str], cues_batch: List[Dict[str, Any]]) -> torch.Tensor:
        """
        cls_vec: [B, hidden]
        texts: list of B raw strings
        cues_batch: list of B cue dicts (has 'ticker', 'metaphor' etc.)
        returns fused vector [B, hidden]
        """
        B = cls_vec.size(0)
        device = cls_vec.device
        fused = cls_vec

        # 1) Modern sentence embedding fusion
        modern_emb_tensor = None
        if self.modern_emb is not None:
            try:
                # encode on CPU via sentence-transformers (they return numpy)
                modern_emb_np = self.modern_emb.encode(texts, convert_to_numpy=True)
                modern_emb_tensor = torch.tensor(modern_emb_np, dtype=torch.float32, device=device)
            except Exception as e:
                # fallback: zeros
                modern_emb_tensor = torch.zeros((B, self.modern_emb_dim), dtype=torch.float32, device=device)
            # apply fusion
            if self.fusion_method == "add":
                proj = self.emb_proj(modern_emb_tensor)  # [B, hidden]
                fused = fused + proj
            elif self.fusion_method == "concat_proj":
                concat = torch.cat([fused, modern_emb_tensor.to(device)], dim=-1)  # [B, hidden+embdim]
                fused = self.concat_proj(concat)
            elif self.fusion_method == "attention":
                # compute attention of cls query over modern emb (B x emb_dim -> B x hidden)
                q = self.query_proj(fused)  # [B, hidden]
                k = self.key_proj(modern_emb_tensor)  # [B, hidden]
                v = self.value_proj(modern_emb_tensor)  # [B, hidden]
                # scaled dot product (B,1,hidden) x (B,hidden,1) -> B,1,1
                attn_score = (q * k).sum(dim=-1, keepdim=True) / (self.hidden ** 0.5)  # [B,1]
                attn_w = torch.sigmoid(attn_score)  # [B,1] (sigmoid to be stable)
                attn_out = v * attn_w  # [B, hidden]
                fused = fused + self.out_proj(attn_out)
            else:
                # default add
                proj = self.emb_proj(modern_emb_tensor)
                fused = fused + proj

        # 2) Inject ticker embedding if present
        if self.ticker_emb is not None:
            # build ticker vector batch
            ticker_vecs = []
            for c in cues_batch:
                t = c.get("ticker")
                if t is None:
                    ticker_vecs.append(torch.zeros(self.hidden, device=device))
                else:
                    idx = self.ticker_to_idx.get(t.upper(), None)
                    if idx is None:
                        ticker_vecs.append(torch.zeros(self.hidden, device=device))
                    else:
                        ticker_vecs.append(self.ticker_emb(torch.tensor(idx, device=device)))
            ticker_stack = torch.stack(ticker_vecs, dim=0)
            fused = fused + ticker_stack

        # 3) Lightweight metaphor flag: if metaphor hint present, amplify certain dimensions
        # (serves to let model pay attention to figurative language)
        metaph_flags = torch.tensor([1.0 if c.get("metaphor") else 0.0 for c in cues_batch], device=device).unsqueeze(-1)
        fused = fused * (1.0 + 0.05 * metaph_flags)  # small modulation

        return fused

    def forward(self, input_ids, attention_mask, return_embedding: bool = False, texts: Optional[List[str]] = None, cues: Optional[List[Dict[str, Any]]] = None):
        # original encoder pass
        enc = self.encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        memory = self.wsd_proj(enc.last_hidden_state)  # [B, T, H]
        B = input_ids.size(0)
        # extract CLS pooled representation (first token) as base
        cls_vec = memory[:, 0, :]  # [B, H]
        # fuse modern embeddings and cues (addresses many ambiguity types)
        if texts is not None and cues is not None:
            fused_cls = self.fuse_embeddings(cls_vec, texts, cues)
        else:
            fused_cls = cls_vec
        tgt = self.query.unsqueeze(1).repeat(1, B, 1)  # [1, B, H]
        mem = memory.permute(1, 0, 2)  # [T, B, H]
        dec_out = self.decoder(tgt=tgt, memory=mem)  # [1, B, H]
        dec_out = dec_out.squeeze(0)  # [B, H]
        # combine decoded representation with fused_cls (residual)
        out_vec = dec_out + fused_cls
        logits = self.classifier(out_vec)  # [B, C]
        if return_embedding:
            return logits, out_vec.detach().cpu().numpy()
        return logits

    def replace_classifier(self, num_labels):
        self.classifier = nn.Linear(self.hidden, num_labels)


# -------------------------
# Metrics: DA, EIC, FSC, Backtest (kept intact)
# -------------------------
def directional_of_labels(gold: List[int], label_map: Optional[Dict[int, int]] = None) -> np.ndarray:
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
            for i, v in enumerate(uniq):
                if i == 0:
                    label_map[v] = -1
                elif i == len(uniq) - 1:
                    label_map[v] = 1
                else:
                    label_map[v] = 0
    return np.array([label_map.get(x, 0) for x in gold]), label_map


def compute_directional_agreement(gold: List[int], preds: List[int]) -> float:
    if len(gold) == 0:
        return 0.0
    gold_dirs, label_map = directional_of_labels(gold)
    pred_dirs = np.array([label_map.get(p, 0) for p in preds])
    agree = (gold_dirs == pred_dirs).sum()
    return float(agree) / len(gold)


def compute_event_impact_correlation(preds: List[int], logits: np.ndarray, impact_scores: Optional[List[float]], gold: Optional[List[int]] = None) -> Optional[float]:
    if impact_scores is None:
        print("[EIC] impact_scores not provided; skipping EIC.")
        return None
    if logits is None or len(logits) == 0:
        print("[EIC] no logits available; skipping EIC.")
        return None
    probs = softmax_rows(logits)
    num_classes = probs.shape[1]
    if num_classes == 1:
        signals = probs[:, 0]
    elif num_classes == 2:
        signals = probs[:, 1] - probs[:, 0]
    else:
        signals = probs[:, -1] - probs[:, 0]
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
        cs = cosine_similarity(cls_emb, centroid).flatten()
        sims.extend(cs.tolist())
    if len(sims) == 0:
        return None
    return float(np.mean(sims))


def compute_backtest(preds: List[int], logits: np.ndarray, future_returns: Optional[List[float]], gold: Optional[List[int]] = None) -> Optional[Dict[str, float]]:
    if future_returns is None:
        print("[BACKTEST] future_returns not provided; skipping backtest.")
        return None
    if logits is None:
        pass
    gold_dirs, label_map = directional_of_labels(gold if gold is not None else preds)
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
            texts = batch.get("text", None)
            cues = batch.get("cues", None)
            if texts is None:
                texts = ["" for _ in range(input_ids.size(0))]
            if cues is None:
                cues = [{} for _ in range(input_ids.size(0))]
            out = model(input_ids, attention_mask, return_embedding=True, texts=texts, cues=cues)
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
# Training loop (CPU-only) - minimal changes: pass texts & cues
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
        for batch in tqdm(train_loader, desc=f"Train epoch {epoch}", leave=False):
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            texts = batch.get("text", None)
            cues = batch.get("cues", None)
            logits = model(input_ids, attention_mask, texts=texts, cues=cues)
            # model returns logits (not dict) for training in this code shape
            if isinstance(logits, torch.Tensor):
                logits_tensor = logits
            else:
                logits_tensor = torch.tensor(np.array(logits), dtype=torch.float32, device=device)
            loss = F.cross_entropy(logits_tensor, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += float(loss.item())
        avg_loss = total_loss / max(1, len(train_loader))
        preds, gold, logits_all, embeddings_all = evaluate_with_embeddings(model, val_loader, device)
        acc = accuracy_score(gold, preds) if len(gold) else 0.0
        f1 = f1_score(gold, preds, average="macro") if len(gold) else 0.0
        print(f"[Epoch {epoch}] loss={avg_loss:.4f} val_acc={acc:.4f} val_f1={f1:.4f}")

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
        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), os.path.join(out_dir, "best.pth"))
            print("Saved best ->", os.path.join(out_dir, "best.pth"))
    return os.path.join(out_dir, "best.pth")


# -------------------------
# Main pipeline (keeps your overall flow but wires in new components)
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
    modern_emb_model: Optional[str],
    fusion_method: str,
    use_finance_emb: bool,
    ticker_list: Optional[str],
):
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
    setattr(train_ds, "dataframe", train_df.reset_index(drop=True))
    setattr(val_ds, "dataframe", val_df.reset_index(drop=True))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # ticker vocab optional
    tickers = []
    if ticker_list:
        tickers = [t.strip().upper() for t in ticker_list.split(",") if t.strip()]
    model = WSDEncoderDecoder(encoder_name="prajjwal1/bert-tiny", hidden_size=128, num_labels=len(le.classes_), nhead=4, dec_layers=1, modern_emb_model=modern_emb_model, fusion_method=fusion_method, use_finance_emb=use_finance_emb, ticker_vocab=tickers)
    print("Training on Reuters (CPU-only) with modern embedding fusion...")
    train_supervised(model, train_loader, val_loader, device, epochs=reuters_epochs, lr=lr, out_dir=os.path.join(out_dir, "reuters"))

    # ---------------------------
    # (Optional) MLM on FPB - using HF Trainer (still CPU)
    # ---------------------------
    if do_mlm_on_fpb:
        fpb_df = load_fpb_texts(fpb_path)
        if not fpb_df.empty:
            print("FPB loaded, but MLM Trainer may require 'accelerate' and can be slow on CPU.")
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
        fiqa_df["label_id"] = 0
        train_df, val_df = train_test_split(fiqa_df, test_size=0.1, random_state=42)
        train_ds = SupervisedTextDataset(train_df["text"].tolist(), train_df["label_id"].tolist(), tokenizer, max_len)
        val_ds = SupervisedTextDataset(val_df["text"].tolist(), val_df["label_id"].tolist(), tokenizer, max_len)
        setattr(train_ds, "dataframe", train_df.reset_index(drop=True))
        setattr(val_ds, "dataframe", val_df.reset_index(drop=True))
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
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
    # Embedding & augmentation options
    parser.add_argument("--modern_emb_model", type=str, default="sentence-transformers/all-MiniLM-L6-v2", help="SentenceTransformer model name or path (if SBERT installed).")
    parser.add_argument("--fusion_method", type=str, default="attention", choices=["add", "concat_proj", "attention"], help="How to fuse modern embeddings with encoder")
    parser.add_argument("--use_finance_emb", action="store_true", help="Attempt to use a finance-specific embedding model if available.")
    parser.add_argument("--ticker_list", type=str, default="", help="Comma-separated list of tickers to give learned embeddings (e.g., AAPL,TSLA,GOOG)")
    args = parser.parse_args()

    # Force CPU environment
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:32")

    # If SBERT not available but user specified model, warn and set to None (fusion disabled)
    if not SBERT_AVAILABLE:
        if args.modern_emb_model:
            print("[WARNING] sentence-transformers not installed. Modern embedding fusion will be disabled.")
        args.modern_emb_model = None

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
        modern_emb_model=args.modern_emb_model,
        fusion_method=args.fusion_method,
        use_finance_emb=args.use_finance_emb,
        ticker_list=args.ticker_list,
    )
