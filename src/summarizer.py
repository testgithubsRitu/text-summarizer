"""Abstractive summarization with a Hugging Face Transformer.

Short text  -> summarised directly.
Long text   -> map-reduce: summarise each chunk, join the partial summaries, and repeat until the
               result fits in one model window; then produce the final summary.
"""
from src.chunking import chunk_text, count_tokens

DEFAULT_MODEL = "sshleifer/distilbart-cnn-12-6"  # distilled BART fine-tuned on CNN/DailyMail
MAX_INPUT_TOKENS = 900  # a little under the 1024-token window to leave room for special tokens


class Summarizer:
    def __init__(self, model_name=DEFAULT_MODEL):
        from transformers import pipeline  # imported lazily (slow import)

        self.pipe = pipeline("summarization", model=model_name)
        self.tokenizer = self.pipe.tokenizer

    def _summarize_chunk(self, text, max_length, min_length):
        n = count_tokens(text, self.tokenizer)
        if n < 40:  # too short to summarise meaningfully
            return text
        max_length = max(10, min(max_length, n // 2))
        min_length = min(min_length, max_length - 1)
        out = self.pipe(text, max_length=max_length, min_length=min_length, do_sample=False, truncation=True)
        return out[0]["summary_text"].strip()

    def summarize(self, text, max_length=130, min_length=30, max_rounds=4):
        text = text.strip()
        if not text:
            return ""
        for _ in range(max_rounds):
            chunks = chunk_text(text, self.tokenizer, MAX_INPUT_TOKENS)
            if len(chunks) == 1:
                return self._summarize_chunk(chunks[0], max_length, min_length)
            text = " ".join(self._summarize_chunk(c, max_length, min_length) for c in chunks)
        return self._summarize_chunk(text, max_length, min_length)
