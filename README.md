# Patent LLMS, Datasets and Tokenizers

A collection of Jupyter notebooks for training patent LLMS, producing datasets for their training and domain-specific tokenizers.

---

## Example installation

```
git clone https://github.com/mhurhangee/patent-llms.git
cd patent-llms
uv venv
source .venv/bin/activate   # on Windows: .venv\Scripts\activate
uv install
uv run jupyter notebook # Or run in IDE 
```

---

## Hugging Face Datasets and models

All the datasets and models are available on my HuggingFace.

- https://huggingface.co/datasets/mhurhangee/
- https://huggingface.co/mhurhangee

---

## Structure

The notebooks cover the following.

- `patBERTa/` - notebooks for training a BERT model on patent domain
- `misc/` - misc notebooks and tests

---

## `patBERTAa/`
patBERT is a RoBERTa-style masked language model pre-trained on European Patent Office independent claims (Claim 1) text. 

The goal is to create a domain-base model for multiple downstream patent NLP tasks, starting with CPC Section classification (A–H). 

### `/0-scraping[...].ipynb`

- **Task:** Scraping the EPO publication server for granted patents and converting XML format into JSON 

```json
{
  "pn" : string,      //patent number
  "c" : {             // Claim
    "1" : string,     //Claim number and claim text
    "2" : string,
    etc...
  },
  "cpc" : array of strings
}
```

- **Data**: Scraping all patents after 20210915.  Before then B1 patents did not have CPC codes.  
- **Size**: Approx ~400,000 claims
- **Output:** Processing claims a JSONL files by week.

### `/1-corpus.ipynb`

- **Task**: Turning the scraped data into a corpus, normalising and processing.
- **Cleaning:** Unicode NFKC, whitespace normalization, reference numeral removal/replacement, length filters.
- **Deduplication:** Exact dedupe applied.
- **Output:** Line-separated corpus of granted claim 1 and all claims for MLM training.

### `/2-tokenizer.ipynb`

- **Task:** Creating a BPE tokenizer on patent claims and analysing
- **Sweep**: On small indep claims, shows 16k good compromise on avg tokens vs paramas, little improvement after 16k.
- **Type:** Byte-level BPE (RoBERTa-compatible), vocab size 8k trained for testing on ind claims (quick), and 16k trained all on claims.
- **Saved:** HF format for downstream tasks.
- **Results:** Performance (avg tokens) plateaus after 16k.  8k choosen to minimise memory for trial runs.
- **Output**: EP patent claim domain specific tokenizer with 8k vocab size.

### `/3-mlm-training.ipynb`

- **Task**: Initial training of patBERTa on local GPU.
- **Nano config for now:**  
  - hidden: 128  
  - layers: 2  
  - heads: 2  
  - FFN: 512  
  - max_position_embeddings = 514 (to allow longer contexts later)  
- **SEQ_LEN:** 128 for current runs; longer contexts deferred (until analysis on claims).
- **Output**: https://huggingface.co/mhurhangee/patroberta-mlm-sl128-v8000

### `/4-cpc-dataset.ipynb`

- **Task**: Create HF dataset of claim 1 and CPC section (A-HY) for eval.
- **Preprocessing**: Same normalisation and cleaning process applied to claims before tokenizer applied to claims.
- **Output**: CSV of claim 1 and CPC as dataset

### `/5-cpc-analysis.ipynb`

- **Task**: Analyze CPC and claim dataset
- **Result**: Average token length ~126 and 95% truncation.  Likely `SEQ_LEN` of 128 too short.  However, possible it is fine as claims typically start with known features which is may be possible to classify from.

### `/6-cpc-fine-tune.ipynb`

- **Task**: Fine-tune MLM model fro to classification task such that can assess performance beyond loss.
- **Result**: Fine-tuned model: ~86% accuracy.
- **Output**: Finetuned classification model.

### `/7-baselines.ipynb`

- **Task**: Establish baselines to compare patBERTa classification against. TF-IDF and distRoberta chosen.
- **Results:** TD-IDF ~ 79% accurate with 50k features; and fine tuned distRoberta ~89%. Therefore, patBERTa model (1.5mil param) better than bag of words, but almost comparable to a model with 82 mil params!

### `/8-mlm-training-512.ipynb`

- **Task**: Training at 512 SEQ length instead of 128.
- **Result**: Much, much too slow on local GPU.
- **Output**: Decision o adopt a curriculum approach 128 -> 256 -> 512 and train on Kaggle GPUs.

### `/9-curr-mlm-pretraining-sl[]-v8000.ipynb`

- **Task**: curriculum training of patBERTa to increase SEQ_LEN.
- **Input**: Increase input data to all claims `mhurhangee/ep-patent-all-claims`
- **Training regime**: 8 epoch @ 128, 4 epoch @ 256, and 2 epoch @ 512
- **Outputs**: 3 models.  https://huggingface.co/mhurhangee/patroberta-mlm-sl128-v8000, https://huggingface.co/mhurhangee/patroberta-mlm-sl256-v8000, https://huggingface.co/mhurhangee/patroberta-mlm-sl512-v8000

### `/10-eval-curr.ipynb` - **WIP**
- **Task**: Using similar approach to `6-cpc-fine-tune.ipynb` fine tune each of the stages at curr 128, 256 and 512 sequence length to see impact of curriculum.

---

### `misc/scraping`

- Scraping EPO publication server for granted patents.

### `misc/preprocessing`

- Extracting claims as JSON from EPO XML
- Creating a corpus of claims from the JSON and uploading to HF
- Extracting CPC data from EPO's RDF for text classification

### `misc/tokenizer`

- Training tokenizers on patent claims and comparing them (vocab size)

---

