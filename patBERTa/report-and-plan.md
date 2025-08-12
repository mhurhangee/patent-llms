# patBERT (RoBERTa-style) — Updated End-to-End Plan

## **Summary**
patBERT is a RoBERTa-style masked language model pre-trained on European Patent Office independent claims (Claim 1) text.  
The goal is to create a domain-specific base model for multiple downstream patent NLP tasks, starting with CPC Section classification (A–H).  
We’ve completed the full data pipeline (cleaning, tokenization, pretraining, evaluation), established baselines (TF-IDF, DistilRoBERTa), and set up a reproducible process to measure the benefit of domain pretraining.

---

## 1 Data Preparation 

### Independent claims corpus - `/1-corpus.ipynb`
- **Source:** ~400k Claim 1 texts from EPO XML, with associated CPC codes - see `0-scraped/`
- **Cleaning:** Unicode NFKC, whitespace normalization, reference numeral removal/replacement, length filters.
- **Deduplication:** Exact dedupe applied.
- **Splits:** Train/val/test — 98/1/1 for pretraining corpus.
- **Output:** Line-separated corpus files for MLM training - see `1-corpus-ind-claims/`

- **Larger corpus:** A larger corpus using dependent claims - see `1-corpus-all-claims/`

- **Reuse:** Reuse same cleaning for classification dataset creation so tokenization matches MLM training.

---

## 2 Tokenizer - `2-tokenizer.ipynb`
- **Sweep**: On small indp claims, shows 16k good compromise on avg tokens vs paramas, little improvement after 16k.
- **Type:** Byte-level BPE (RoBERTa-compatible), vocab size 8k trained for testing on ind claims (quick), and 16k trained all on claims.
- **Trained:** On training split only.
- **Saved:** HF format for downstream tasks.

> Performance (avg tokens) plateaus after 16k.  8k choosen to minimise memory.

---

## 3 Model Config
- **Nano config for now:**  
  - hidden: 128  
  - layers: 2  
  - heads: 2  
  - FFN: 512  
  - max_position_embeddings = 514 (to allow longer contexts later)  
- **SEQ_LEN:** 128 for current runs; longer contexts deferred.

> Train from start with `max_position_embeddings=514` to avoid reinitialization later.

---

## 4 MLM Pretraining
- **Objective:** MLM (15% mask rate), dynamic masking.
- **Current run:** SEQ_LEN=128, 2 epochs.
- **Optimizer:** AdamW, lr=5e-4, weight decay 0.01, warmup ratio=0.06.
- **Batch size:** GPU-fit with monitoring; FP16 enabled.

> Pretrained checkpoint (`patroberta-mlm-128`) ready for fine-tuning.

---

## 5 CPC Section Classification Dataset
- **Step 1:** Extract Claim 1 + CPC codes from JSONL files.  
  Reduce to top-level CPC letter (A–H).  
  Keep only samples with a single valid section.
- **Stats:** ~216k kept; balanced splits train/val/test.
- **Step 2:** Tokenize with domain tokenizer, MAX_LEN=128.  
  Result: ~96% truncation rate → will need longer context for better recall.

> Use this dataset as the primary downstream benchmark.

---

## 6 Downstream Fine-tuning
- **Model:** `RobertaForSequenceClassification` (from patBERT MLM checkpoint).
- **Task:** 8-way classification (A–H).
- **Metrics:** Accuracy, macro-F1, weighted-F1.
- **Outcome so far:** ~86% accuracy at 2 epochs, MAX_LEN=128 → beats TF-IDF baseline (~79%) similar to distilRoBERTa (~88-89%).

> Classification performance is of my trained model is comparable to distilRoBERTa for less computional effort (PRETRAINING & FINETUNING < FINETUNING ROBERTA)

---

## 7 Baselines
- **TF-IDF + Logistic Regression:** 79% accuracy (max_features=50k).
- **DistilRoBERTa Fine-tune:** 88-89% accuracy (truncation at 128)
- **Decision:** Store baseline metrics alongside patBERT results for ongoing comparison.

> Need to increase the SEQ_LEN, and  512 seems logical.  Reassess performance at this level.

> Assess curriculum approach 128 -> 256 -> 512.

---

## 8

---

## **8) Next Steps**
1. **Longer Context Training** — Move MLM + classification to 256/512 seq length to reduce truncation.
2. **Baseline Expansion** — Evaluate more complex/multilabel CPC setups.
3. **Pretrain Scaling** — Increase model size, dataset size, and epochs.
4. **HPO** — Light hyperparameter sweeps for MLM and downstream tasks.
5. **Ops** — Ensure reproducibility (fixed seeds, version logging, artifact hashes).

---
