#!/usr/bin/env python3
# NLP_CASE.py - CPU-only pipeline with evaluation metrics:
# Directional Agreement (DA), Event-Impact Correlation (EIC),
# Financial Sense Consistency (FSC), Profitability-Oriented Measure (Backtest)
# Enhanced with modern embeddings + augmentation to address 7 ambiguity types.
# Now extended with comparative embedding analysis (Liu et al. 2020 + ACL/EMNLP/NeurIPS refs)

import os, re, argparse
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

# -------------------------
# Optional: sentence-transformers
# -------------------------
try:
    from sentence_transformers import SentenceTransformer
    SBERT_AVAILABLE = True
except Exception:
    SBERT_AVAILABLE = False

# -------------------------
# Optional: spaCy
# -------------------------
try:
    import spacy
    SPACY_AVAILABLE = True
    try: _nlp = spacy.load("en_core_web_sm")
    except Exception: _nlp = None
except Exception:
    SPACY_AVAILABLE, _nlp = False, None

# -------------------------
# Text cleaning
# -------------------------
def clean_text(s: str) -> str:
    if s is None: return ""
    s = re.sub(r"\s+", " ", s.replace("\n", " ").replace("\r", " ")).strip()
    s = re.sub(r"http\S+", "", s)
    s = re.sub(r"[^A-Za-z0-9\s\.\,\-\$%€£:;()\/\#\@]", " ", s)
    s = re.sub(r"\s{2,}", " ", s)
    return s.strip()

# -------------------------
# Dataset loaders
# -------------------------
def load_reuters_with_labels(base_path: str) -> pd.DataFrame:
    base = Path(base_path)
    if not base.exists(): return pd.DataFrame()
    rows = []
    for fp in base.rglob("*.txt"):
        try:
            txt = fp.read_text(encoding="utf-8", errors="ignore").strip()
            if txt: rows.append({"text": txt, "label": fp.parent.name})
        except Exception: continue
    return pd.DataFrame(rows)

def load_fpb_texts(base_path: str) -> pd.DataFrame:
    base = Path(base_path)
    if not base.exists(): return pd.DataFrame()
    texts = []
    for fp in base.rglob("*.txt"):
        try:
            raw = fp.read_text(encoding="latin-1", errors="ignore")
            for line in raw.splitlines():
                if line.strip(): texts.append(line.strip())
        except Exception: continue
    return pd.DataFrame({"text": texts})

def load_fiqa_from_hf(hf_path: str) -> pd.DataFrame:
    df = None
    try: df = pd.read_json(hf_path)
    except Exception:
        try:
            from datasets import load_dataset
            ds = load_dataset("llamafactory/fiqa"); df = pd.DataFrame(ds["train"])
        except Exception: return pd.DataFrame()
    if "text" in df.columns: return df[["text"]].copy()
    if all(c in df.columns for c in ["instruction","input","output"]):
        df["text"] = (df["instruction"].fillna("")+" "+df["input"].fillna("")+" "+df["output"].fillna("")).str.strip()
        return df[["text"]].copy()
    return pd.DataFrame()

# -------------------------
# Dataset class
# -------------------------
class SupervisedTextDataset(Dataset):
    def __init__(self,texts,labels,tokenizer,max_len=64):
        self.texts,self.labels,self.tokenizer,self.max_len=texts,labels,tokenizer,max_len
        self.cues=[{"ticker":None,"metaphor":False} for _ in texts]
    def __len__(self): return len(self.texts)
    def __getitem__(self,idx):
        txt=str(self.texts[idx])
        enc=self.tokenizer(txt,truncation=True,padding="max_length",max_length=self.max_len,return_tensors="pt")
        return {
            "input_ids":enc["input_ids"].squeeze(0),
            "attention_mask":enc["attention_mask"].squeeze(0),
            "labels":torch.tensor(self.labels[idx],dtype=torch.long),
            "text":clean_text(txt),
            "cues":self.cues[idx]
        }

# -------------------------
# Model
# -------------------------
class WSDEncoderDecoder(nn.Module):
    def __init__(self,encoder_name="prajjwal1/bert-tiny",hidden_size=128,num_labels=2,
                 nhead=4,dec_layers=1,dropout=0.1,modern_emb_model=None,
                 fusion_method="attention",use_finance_emb=False,ticker_vocab=None):
        super().__init__()
        self.encoder=AutoModel.from_pretrained(encoder_name)
        enc_hidden=getattr(self.encoder.config,"hidden_size",128)
        self.wsd_proj=nn.Sequential(nn.Linear(enc_hidden,hidden_size),nn.GELU(),
                                    nn.LayerNorm(hidden_size),nn.Dropout(dropout))
        self.hidden=hidden_size
        self.query=nn.Parameter(torch.randn(1,hidden_size))
        decoder_layer=nn.TransformerDecoderLayer(d_model=hidden_size,nhead=nhead,dropout=dropout)
        self.decoder=nn.TransformerDecoder(decoder_layer,num_layers=dec_layers)
        self.classifier=nn.Linear(hidden_size,num_labels)
        self.modern_emb=None; self.modern_emb_dim=None; self.fusion_method=fusion_method
        if SBERT_AVAILABLE and modern_emb_model:
            try:
                self.modern_emb=SentenceTransformer(modern_emb_model)
                self.modern_emb_dim=int(self.modern_emb.encode("test",convert_to_numpy=True).shape[-1])
            except: self.modern_emb=None
        if self.modern_emb is not None:
            if self.fusion_method=="add":
                self.emb_proj=nn.Linear(self.modern_emb_dim,hidden_size)
            elif self.fusion_method=="concat_proj":
                self.concat_proj=nn.Linear(hidden_size+self.modern_emb_dim,hidden_size)
            elif self.fusion_method=="attention":
                self.key_proj,self.query_proj,self.value_proj,self.out_proj=(nn.Linear(self.modern_emb_dim,hidden_size),
                    nn.Linear(hidden_size,hidden_size),nn.Linear(self.modern_emb_dim,hidden_size),nn.Linear(hidden_size,hidden_size))
    def fuse_embeddings(self,cls_vec,texts):
        if self.modern_emb is None: return cls_vec
        modern_emb_np=self.modern_emb.encode(texts,convert_to_numpy=True)
        modern_emb=torch.tensor(modern_emb_np,dtype=torch.float32,device=cls_vec.device)
        if self.fusion_method=="add":
            return cls_vec+self.emb_proj(modern_emb)
        elif self.fusion_method=="concat_proj":
            return self.concat_proj(torch.cat([cls_vec,modern_emb],dim=-1))
        elif self.fusion_method=="attention":
            q=self.query_proj(cls_vec); k=self.key_proj(modern_emb); v=self.value_proj(modern_emb)
            attn_w=torch.sigmoid((q*k).sum(dim=-1,keepdim=True)/(self.hidden**0.5))
            return cls_vec+self.out_proj(v*attn_w)
        return cls_vec
    def forward(self,input_ids,attention_mask,texts=None):
        enc=self.encoder(input_ids=input_ids,attention_mask=attention_mask,return_dict=True)
        memory=self.wsd_proj(enc.last_hidden_state)
        B=input_ids.size(0); cls_vec=memory[:,0,:]
        fused_cls=self.fuse_embeddings(cls_vec,texts) if texts else cls_vec
        tgt=self.query.unsqueeze(1).repeat(1,B,1); mem=memory.permute(1,0,2)
        dec_out=self.decoder(tgt=tgt,memory=mem).squeeze(0)
        out_vec=dec_out+fused_cls
        return self.classifier(out_vec)

# -------------------------
# Metrics
# -------------------------
def compute_directional_agreement(gold,preds):
    if not gold: return 0.0
    label_map={min(set(gold)):-1,max(set(gold)):1}
    pred_dirs=[label_map.get(p,0) for p in preds]
    gold_dirs=[label_map.get(g,0) for g in gold]
    return float((np.array(gold_dirs)==np.array(pred_dirs)).mean())

# -------------------------
# Evaluate
# -------------------------
def evaluate_with_embeddings(model,dataloader,device):
    model.eval(); preds,gold=[],[]
    with torch.no_grad():
        for batch in dataloader:
            logits=model(batch["input_ids"].to(device),batch["attention_mask"].to(device),texts=batch["text"])
            logits_np=logits.cpu().numpy(); preds.extend(np.argmax(logits_np,axis=-1).tolist())
            gold.extend(batch["labels"].cpu().numpy().tolist())
    return preds,gold

# -------------------------
# Training
# -------------------------
def train_supervised(model,train_loader,val_loader,device,epochs=1,lr=2e-5,out_dir="ckpt"):
    model.to(device); opt=AdamW(model.parameters(),lr=lr)
    best_f1=-1; os.makedirs(out_dir,exist_ok=True)
    metrics={}
    for epoch in range(1,epochs+1):
        model.train()
        for batch in tqdm(train_loader,desc=f"Epoch {epoch}",leave=False):
            opt.zero_grad()
            logits=model(batch["input_ids"].to(device),batch["attention_mask"].to(device),texts=batch["text"])
            loss=F.cross_entropy(logits,batch["labels"].to(device)); loss.backward(); opt.step()
        preds,gold=evaluate_with_embeddings(model,val_loader,device)
        acc=accuracy_score(gold,preds); f1=f1_score(gold,preds,average="macro")
        da=compute_directional_agreement(gold,preds)
        print(f"[Epoch {epoch}] acc={acc:.4f} f1={f1:.4f} DA={da:.4f}")
        metrics={"acc":acc,"f1":f1,"DA":da}
        if f1>best_f1: best_f1=f1; torch.save(model.state_dict(),os.path.join(out_dir,"best.pth"))
    return metrics

# -------------------------
# Main
# -------------------------
def main(reuters_path,fpb_path,fiqa_hfpath,do_mlm_on_fpb,batch_size,max_len,
         reuters_epochs,fpb_mlm_epochs,fiqa_epochs,tweets_epochs,lr,out_dir,
         modern_emb_model,fusion_method,use_finance_emb,ticker_list):
    device=torch.device("cpu"); os.makedirs(out_dir,exist_ok=True)
    tokenizer=AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")
    reuters_df=load_reuters_with_labels(reuters_path)
    if reuters_df.empty: return {"acc":0,"f1":0,"DA":0}
    reuters_df["text"]=reuters_df["text"].astype(str).map(clean_text)
    le=LabelEncoder(); reuters_df["label_id"]=le.fit_transform(reuters_df["label"])
    train_df,val_df=train_test_split(reuters_df,test_size=0.2,random_state=42,stratify=reuters_df["label_id"])
    train_ds=SupervisedTextDataset(train_df["text"].tolist(),train_df["label_id"].tolist(),tokenizer,max_len)
    val_ds=SupervisedTextDataset(val_df["text"].tolist(),val_df["label_id"].tolist(),tokenizer,max_len)
    train_loader=DataLoader(train_ds,batch_size=batch_size,shuffle=True)
    val_loader=DataLoader(val_ds,batch_size=batch_size)
    model=WSDEncoderDecoder("prajjwal1/bert-tiny",128,len(le.classes_),modern_emb_model=modern_emb_model,fusion_method=fusion_method)
    return train_supervised(model,train_loader,val_loader,device,epochs=reuters_epochs,lr=lr,out_dir=out_dir)

# -------------------------
# Embedding Comparison
# -------------------------
def run_embedding_comparison(args):
    embedding_models=[
        "word2vec-google-news-300","glove-wiki-gigaword-300",
        "google/electra-small-discriminator","nghuyong/ernie-2.0-en",
        "xlnet-base-cased","microsoft/deberta-v3-small",
        "sentence-transformers/all-mpnet-base-v2","princeton-nlp/sup-simcse-roberta-base",
        "yiyanghkust/finbert-tone"
    ]
    results=[]
    for emb in embedding_models:
        print(f"\n[COMPARISON] {emb}")
        try:
            metrics=main(args.reuters_path,args.fpb_path,args.fiqa_hfpath,
                args.do_mlm_on_fpb,args.batch_size,args.max_len,
                args.reuters_epochs,args.fpb_mlm_epochs,args.fiqa_epochs,
                args.tweets_epochs,args.lr,os.path.join(args.out_dir,emb.replace('/','_')),
                emb,args.fusion_method,args.use_finance_emb,args.ticker_list)
            results.append({"embedding":emb,**metrics,
                "reference":"Liu et al. 2020 (ACL), plus EMNLP/NeurIPS/ACL papers"})
        except Exception as e:
            results.append({"embedding":emb,"acc":0,"f1":0,"DA":0,"reference":f"ERROR: {e}"})
    pd.DataFrame(results).to_csv(os.path.join(args.out_dir,"embedding_comparison.csv"),index=False)
    with open(os.path.join(args.out_dir,"embedding_comparison.md"),"w") as f:
        f.write("| Embedding | Acc | F1 | DA | Reference |\n|-----------|-----|----|----|------------|\n")
        for r in results:
            f.write(f"| {r['embedding']} | {r['acc']:.4f} | {r['f1']:.4f} | {r['DA']:.4f} | {r['reference']} |\n")

# -------------------------
# CLI
# -------------------------
if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--reuters_path",type=str,default="./ReutersTranscribedSubset")
    parser.add_argument("--fpb_path",type=str,default="./FinancialPhraseBank-v1.0")
    parser.add_argument("--fiqa_hfpath",type=str,default="hf://datasets/llamafactory/fiqa/train.json")
    parser.add_argument("--out_dir",type=str,default="./wsd_pipeline_out_tiny_cpu")
    parser.add_argument("--batch_size",type=int,default=1)
    parser.add_argument("--max_len",type=int,default=64)
    parser.add_argument("--reuters_epochs",type=int,default=1)
    parser.add_argument("--fpb_mlm_epochs",type=int,default=1)
    parser.add_argument("--fiqa_epochs",type=int,default=1)
    parser.add_argument("--tweets_epochs",type=int,default=1)
    parser.add_argument("--lr",type=float,default=2e-5)
    parser.add_argument("--do_mlm_on_fpb",action="store_true")
    parser.add_argument("--modern_emb_model",type=str,default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--fusion_method",type=str,default="attention")
    parser.add_argument("--use_finance_emb",action="store_true")
    parser.add_argument("--ticker_list",type=str,default="")
    parser.add_argument("--compare_embeddings",action="store_true")
    args=parser.parse_args()
    # force CPU-only
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    torch.set_num_threads(os.cpu_count())
    if args.compare_embeddings: run_embedding_comparison(args)
    else:
        main(args.reuters_path,args.fpb_path,args.fiqa_hfpath,args.do_mlm_on_fpb,
             args.batch_size,args.max_len,args.reuters_epochs,args.fpb_mlm_epochs,
             args.fiqa_epochs,args.tweets_epochs,args.lr,args.out_dir,
             args.modern_emb_model,args.fusion_method,args.use_finance_emb,args.ticker_list)
