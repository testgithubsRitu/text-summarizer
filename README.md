# Text Summarizer: Transformer-based Automated Summaries

An automated summary generator in Python using a Hugging Face **Transformer** (DistilBART fine-tuned on
CNN/DailyMail). It summarises short text directly and handles **long documents** with token-aware chunking and
a map-reduce strategy, with a CLI, a Streamlit web UI and a **ROUGE evaluation** script.

## How it works

```
 long text ─► split into sentences ─► pack into chunks ≤ 900 tokens (model's own tokenizer)
          ─► summarise each chunk ─► join partial summaries ─► (repeat if still too long) ─► final summary
```

- **Why chunk?** DistilBART reads at most 1024 tokens. Longer input would be silently truncated.
- **Why tokens, not words?** Transformers read sub-word tokens; counting with the model's tokenizer
  (`src/chunking.py`) keeps every chunk inside the window.
- **Why sentence boundaries?** Cutting mid-sentence degrades summary quality.
- Generation is deterministic (`do_sample=False`), so the same input gives the same summary.

## Quick start

```bash
git clone <your-repo-url> && cd text-summarizer
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python cli.py data/sample_article.txt                  # first run downloads the model (~1.2 GB)
streamlit run app.py                                   # web UI: paste text or upload a .txt file
```

## Evaluation

```bash
python -m src.evaluate data/eval_sample.csv
```

Reads a CSV with `text` and `reference` columns, generates summaries and reports ROUGE-1/2/L F-scores per
row and on average (via `pandas` + `rouge-score`). The included sample has only 2 rows, so treat its numbers as
a demo. For meaningful evaluation use a standard dataset such as CNN/DailyMail.

## Tests

```bash
pytest -q
```

Tests use a fake whitespace tokenizer (no model download) and cover sentence splitting, the chunk token budget,
no text being lost across chunks, oversized sentences, and the ROUGE helper.

## Project structure

```
app.py               Streamlit UI
cli.py               command-line summarizer
src/chunking.py      sentence + token-aware chunking
src/summarizer.py    Hugging Face pipeline + map-reduce
src/evaluate.py      ROUGE evaluation
tests/               unit tests
data/                sample article and tiny eval CSV
```

## Limitations and next steps

- Abstractive models can state things not in the source; verify important summaries.
- DistilBART is tuned for English news-style text; try `facebook/bart-large-cnn` or `google/pegasus-xsum`
  by changing `DEFAULT_MODEL` in `src/summarizer.py`.
- Next: compare models on ROUGE, fine-tune a small model on a custom dataset, add a length-controlled summary option.
