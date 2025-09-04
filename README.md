# **NLP_CASE – Memory-Optimized Financial NLP Pipeline**

This repository contains a **memory-optimized NLP pipeline** designed for training and fine-tuning transformer-based models on **financial datasets** using **≤4GB GPUs**.

The pipeline performs the following tasks:
- Supervised classification on the **Reuters dataset**
- Masked Language Modeling (MLM) on the **Financial PhraseBank (FPB)**
- Fine-tuning on the **FiQA financial QA dataset**
- Optionally integrates **Financial Tweets** via KaggleHub

---

## **📌 Features**
- Uses **BERT-tiny** for efficiency on low-memory GPUs
- Optimized for **≤4GB VRAM** using:
  - Gradient checkpointing
  - Mixed precision training (`torch.amp`)
  - Expandable CUDA memory segments
- Supports multi-dataset fine-tuning
- Includes automatic model checkpoint saving
- Supports **HF datasets** + **local dataset loading**

---

## **📂 Project Structure**
```

NLP\_CASE/
│── NLP\_CASE.py              # Main pipeline script
│── README.md                # Documentation
│── wsd\_pipeline\_out\_tiny/   # Auto-generated model checkpoints
│── datasets/                # Local datasets
│     ├── ReutersTranscribedSubset/
│     ├── FinancialPhraseBank-v1.0/
│     └── fiqa/

````

---

## **📊 Datasets Used**
| **Dataset**          | **Purpose**               | **Source** |
|----------------------|---------------------------|------------|
| Reuters             | Supervised classification | Local path |
| Financial PhraseBank | MLM pretraining           | Local path |
| FiQA                | Financial Q&A fine-tuning | HuggingFace |
| Financial Tweets    | Sentiment analysis        | KaggleHub |

---

## **⚡ Installation**

### **1. Clone the repository**
```bash
git clone https://github.com/<your-username>/NLP_CASE.git
cd NLP_CASE
````

### **2. Create a virtual environment**

```bash
python3 -m venv venv
source venv/bin/activate
```

### **3. Install dependencies**

```bash
pip install torch torchvision torchaudio
pip install transformers accelerate datasets tqdm scikit-learn pandas numpy kagglehub
```

---

## **🚀 Usage**

### **Basic Command**

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:32
python3 NLP_CASE.py \
    --do_mlm_on_fpb \
    --batch_size 1 \
    --max_len 64 \
    --reuters_epochs 1 \
    --fpb_mlm_epochs 1 \
    --fiqa_epochs 1 \
    --tweets_epochs 1
```

---

## **⚙ Command-Line Arguments**

| **Argument**       | **Type** | **Default**                                  | **Description**                   |
| ------------------ | -------- | -------------------------------------------- | --------------------------------- |
| `--reuters_path`   | str      | `./ReutersTranscribedSubset`                 | Reuters dataset path              |
| `--fpb_path`       | str      | `./FinancialPhraseBank-v1.0`                 | Financial PhraseBank dataset path |
| `--fiqa_hfpath`    | str      | `hf://datasets/llamafactory/fiqa/train.json` | FiQA dataset path                 |
| `--out_dir`        | str      | `./wsd_pipeline_out_tiny`                    | Directory for checkpoints         |
| `--batch_size`     | int      | `1`                                          | Batch size for training           |
| `--max_len`        | int      | `64`                                         | Max sequence length               |
| `--reuters_epochs` | int      | `1`                                          | Training epochs for Reuters       |
| `--fpb_mlm_epochs` | int      | `1`                                          | MLM training epochs               |
| `--fiqa_epochs`    | int      | `1`                                          | FiQA fine-tuning epochs           |
| `--tweets_epochs`  | int      | `1`                                          | Financial tweets training         |
| `--lr`             | float    | `2e-5`                                       | Learning rate                     |
| `--do_mlm_on_fpb`  | flag     | `False`                                      | Enable FPB MLM pretraining        |

---

## **📦 Output**

After training, the checkpoints are saved here:

```
wsd_pipeline_out_tiny/
├── reuters/
│     └── best.pth
├── fiqa/
│     └── best.pth
```

---

## **🛠 Troubleshooting**

### **1. MLM Trainer Accelerate Error**

If you see:

```
Using the Trainer with PyTorch requires accelerate>=0.26.0
```

Fix it by:

```bash
pip install --upgrade accelerate
```

### **2. KaggleHub Financial Tweets Warning**

If you get:

```
expected str, bytes or os.PathLike object, not NoneType
```

This means the dataset wasn't downloaded.
You can skip `--tweets_epochs` or manually download the dataset.

### **3. CUDA Out of Memory**

* Reduce `--batch_size`
* Use `prajjwal1/bert-tiny` (already set as default)
* Lower `--max_len`

---

## **📌 Example Run**

```bash
python3 NLP_CASE.py \
    --do_mlm_on_fpb \
    --batch_size 1 \
    --max_len 64 \
    --reuters_epochs 3 \
    --fpb_mlm_epochs 2 \
    --fiqa_epochs 2
```

**Example Output:**

```
[Epoch 1] loss=1.7973 val_acc=0.16 val_macro_f1=0.0459
Saved best -> ./wsd_pipeline_out_tiny/reuters/best.pth
[Epoch 1] loss=0.0000 val_acc=1.0 val_macro_f1=1.0
Saved best -> ./wsd_pipeline_out_tiny/fiqa/best.pth
```
<img width="817" height="142" alt="Screenshot from 2025-09-04 11-45-25" src="https://github.com/user-attachments/assets/cbaf9fad-6f0a-4c97-8b27-ccf64bbb57d7" />


---

## **👤 Author**

**Sanjith Ganesa P**
📧 Email: [cb.en.u4cse22043@cb.students.amrita.edu](mailto:cb.en.u4cse22043@cb.students.amrita.edu)
📍 Amrita Vishwa Vidyapeetham, Coimbatore

```

---
