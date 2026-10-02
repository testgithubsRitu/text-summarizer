"""Token-aware chunking for long documents.

Summarization models have a fixed input window (DistilBART: 1024 tokens). A long article must be
split first. We split on sentence boundaries and pack sentences into chunks that stay under a
token budget, measuring length with the model's own tokenizer (not characters or words).
"""
import re

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def count_tokens(text, tokenizer):
    return len(tokenizer.encode(text, add_special_tokens=False))


def split_sentences(text):
    return [s.strip() for s in _SENTENCE_SPLIT.split(text.strip()) if s.strip()]


def chunk_text(text, tokenizer, max_tokens=900):
    """Pack whole sentences into chunks of at most `max_tokens` tokens.

    A single sentence longer than the budget becomes its own chunk (the model will truncate it).
    """
    chunks, current, current_len = [], [], 0
    for sentence in split_sentences(text):
        n = count_tokens(sentence, tokenizer)
        if current and current_len + n > max_tokens:
            chunks.append(" ".join(current))
            current, current_len = [], 0
        current.append(sentence)
        current_len += n
    if current:
        chunks.append(" ".join(current))
    return chunks
