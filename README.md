# WSD Financial NLP Pipeline 

**Project:** Memory-efficient NLP pipeline for financial text understanding and supervised learning using Tiny BERT-based encoder-decoder models.  
**Author:** Sanjith Ganesa P  
**Date:** 2025  

---

## Table of Contents
1. [Project Overview](#project-overview)  
2. [Features](#features)  
3. [Datasets](#datasets)  
4. [Evaluation Metrics](#evaluation-metrics)  
5. [Environment Setup](#environment-setup)  
6. [Usage](#usage)  
7. [Pipeline Implementation](#pipeline-implementation)  
8. [Output](#output)  

---

## Project Overview

This project implements a memory-optimized NLP pipeline for financial text analysis using **CPU-only training** (suitable for machines with ≤4GB GPU or CPU-only environments). The pipeline supports:

- Supervised classification on **Reuters** financial news.  
- Masked Language Modeling (MLM) fine-tuning on **Financial PhraseBank (FPB)**.  
- Supervised fine-tuning on **FiQA** financial sentiment dataset.  
- Optional integration of financial tweets from Kaggle datasets.  

The model architecture is a **Tiny BERT-based Encoder-Decoder** for word sense disambiguation and classification. The design ensures low memory usage while maintaining performance.

---

## Features

- CPU-only training with `torch.amp` support for mixed precision.  
- Flexible dataset loading and cleaning: Reuters, FPB, FiQA, Kaggle tweets.  
- Supervised training with **cross-entropy loss** and model checkpointing.  
- Evaluation metrics specifically tailored for financial NLP:  
  - **Directional Agreement (DA)**  
  - **Event-Impact Correlation (EIC)**  
  - **Financial Sense Consistency (FSC)**  
  - **Profitability-Oriented Measure (Backtest Metric)**  

---

## Datasets

1. **Reuters Subset** – Financial news labeled by category.  
2. **Financial PhraseBank (FPB)** – Sentences annotated for sentiment.  
3. **FiQA** – Financial question-answer dataset from HuggingFace.  
4. **Financial Tweets** – Optional Kaggle dataset of finance-related tweets.

> Data paths should be provided in the CLI arguments when running the script.

---

## Evaluation Metrics

- **Directional Agreement (DA):** Measures alignment between predicted and true sentiment directions.  
- **Event-Impact Correlation (EIC):** Correlation between events and model-predicted impacts (requires impact_scores).  
- **Financial Sense Consistency (FSC):** Semantic consistency of financial statements post-prediction.  
- **Profitability-Oriented Measure (Backtest):** Evaluates model predictions against actual financial returns (requires future_returns).  

> Missing optional metrics will be skipped during evaluation.

---

## Environment Setup

1. **Clone the repository**:
```bash
git clone https://github.com/your-username/wsd-financial-nlp.git
cd wsd-financial-nlp
````

2. **Create virtual environment**:

```bash
python3 -m venv venv
source venv/bin/activate
```

3. **Install dependencies**:

```bash
pip install -r requirements.txt
```

> Sample `requirements.txt` includes:

```
torch
transformers
datasets
pandas
numpy
scikit-learn
tqdm
kagglehub
```

---

## Usage

**Run the pipeline:**

```bash
python3 NLP_CASE.py \
    --do_mlm_on_fpb \
    --batch_size 1 \
    --max_len 64 \
    --reuters_epochs 1 \
    --fpb_mlm_epochs 1 \
    --fiqa_epochs 1 \
    --tweets_epochs 1
```

**CLI Arguments:**

* `--reuters_path`: Path to Reuters subset.
* `--fpb_path`: Path to Financial PhraseBank.
* `--fiqa_hfpath`: Path/URL to FiQA dataset.
* `--out_dir`: Directory for checkpoints and outputs.
* `--batch_size`: Training batch size.
* `--max_len`: Maximum token length.
* `--reuters_epochs`: Epochs for Reuters supervised training.
* `--fpb_mlm_epochs`: Epochs for FPB MLM fine-tuning.
* `--fiqa_epochs`: Epochs for FiQA fine-tuning.
* `--tweets_epochs`: Epochs for optional financial tweets fine-tuning.
* `--do_mlm_on_fpb`: Flag to enable FPB MLM fine-tuning.

> All computations are forced on CPU for memory efficiency.

---

## Pipeline Implementation

### 1. Text Cleaning

```python
def clean_text(s: str) -> str:
    if s is None: return ""
    s = re.sub(r"\s+", " ", s.replace("\n", " ").replace("\r", " ")).strip()
    s = re.sub(r"http\S+", "", s)
    s = re.sub(r"[^A-Za-z0-9\s\.\,\-\$%€£:;()\/]", " ", s)
    s = re.sub(r"\s{2,}", " ", s)
    return s.strip()
```

### 2. Dataset Loaders

* **Reuters** – Recursive TXT loading with labels from directory structure.
* **FPB** – Load text sentences from `.txt` files.
* **FiQA** – Robust HuggingFace loader with fallback to JSON.
* **Financial Tweets** – Kagglehub loader (optional).

### 3. Dataset Class

```python
class SupervisedTextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=64):
        self.texts, self.labels = texts, labels
        self.tokenizer, self.max_len = tokenizer, max_len
    def __len__(self): return len(self.texts)
    def __getitem__(self, idx):
        enc = self.tokenizer(str(self.texts[idx]), truncation=True,
                             padding="max_length", max_length=self.max_len, return_tensors="pt")
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long)
        }
```

### 4. Model Architecture

* **Encoder:** Tiny BERT (`prajjwal1/bert-tiny`)
* **WSD Projection:** Linear → GELU → LayerNorm → Dropout
* **Decoder:** TransformerDecoder with learned query
* **Classifier:** Linear head for label prediction

```python
class WSDEncoderDecoder(nn.Module):
    def __init__(self, encoder_name="prajjwal1/bert-tiny", hidden_size=128, num_labels=2, nhead=4, dec_layers=1, dropout=0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_name)
        self.encoder.gradient_checkpointing_enable()
        enc_hidden = getattr(self.encoder.config, "hidden_size", 128)
        self.wsd_proj = nn.Sequential(nn.Linear(enc_hidden, hidden_size), nn.GELU(),
                                      nn.LayerNorm(hidden_size), nn.Dropout(dropout))
        self.hidden = hidden_size
        self.query = nn.Parameter(torch.randn(1, hidden_size))
        decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_size, nhead=nhead, dropout=dropout)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=dec_layers)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        enc = self.encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        memory = self.wsd_proj(enc.last_hidden_state)
        B = input_ids.size(0)
        tgt = self.query.unsqueeze(1).repeat(1, B, 1)
        dec_out = self.decoder(tgt=tgt, memory=memory.permute(1,0,2))
        return self.classifier(dec_out.squeeze(0))

    def replace_classifier(self, num_labels):
        self.classifier = nn.Linear(self.hidden, num_labels)
```

### 5. Training & Evaluation

* CPU training using **AdamW optimizer**, **StepLR scheduler**, and **GradScaler**.
* Evaluation outputs **accuracy**, **macro F1**, DA, EIC, FSC, and optional backtest metric.

```python
acc, f1 = evaluate(model, val_loader, device)
print(f"[Epoch {epoch}] loss={avg_loss:.4f} val_acc={acc} val_f1={f1}")
print(f"DA={da_score:.4f} EIC={eic_score} FSC={fsc_score}")
```

### 6. Full Pipeline Execution

```bash
python3 NLP_CASE.py --do_mlm_on_fpb --batch_size 1 --max_len 64 --reuters_epochs 1 --fpb_mlm_epochs 1 --fiqa_epochs 1 --tweets_epochs 1
```

---

## Output

* Model checkpoints saved in `./wsd_pipeline_out_tiny_cpu/`.
* Evaluation metrics logged after each epoch.

Example:

```
[Epoch 1] loss=1.9492 val_acc=0.2000 val_f1=0.1049
DA=0.2400  EIC=N/A  FSC=0.9934
Saved best -> ./wsd_pipeline_out_tiny_cpu/reuters/best.pth
```

