# Patent LLMS, Datasets and Tokenizers

A collection of Jupyter notebooks for training patent LLMS, producing datasets for their training and domain-specific tokenizers.

## Example installation

```
git clone https://github.com/mhurhangee/patent-llms.git
cd patent-llms
uv venv
source .venv/bin/activate   # on Windows: .venv\Scripts\activate
uv install
uv run jupyter notebook # Or run in IDE 
```

## Structure

The notebooks cover the following.

### `/scraping`

- Scraping EPO publication server for granted patents.

### `/preprocessing`

- Extracting claims as JSON from EPO XML
- Creating a corpus of claims from the JSON and uploading to HF
- Extracting CPC data from EPO's RDF for text classification

## `/tokenizer`

- Training tokenizers on patent claims and comparing them (vocab size)

## Hugging Face Datasets

- https://huggingface.co/datasets/mhurhangee/patent-ind-claim-en 
- https://huggingface.co/datasets/mhurhangee/cpc-classifications