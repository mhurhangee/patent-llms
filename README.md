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
- Creating a corpus of claims from the JSON
- Uploading the corpus to a HF dataset

## Hugging Face Datasets

- https://huggingface.co/datasets/mhurhangee/patent-ind-claim-en 