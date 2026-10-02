"""Tests use a fake whitespace tokenizer so they run without downloading any model."""
from src.chunking import chunk_text, count_tokens, split_sentences
from src.evaluate import rouge_table


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        return text.split()


TOK = FakeTokenizer()


def test_split_sentences():
    assert split_sentences("One. Two! Three? Four") == ["One.", "Two!", "Three?", "Four"]


def test_chunks_stay_under_budget_and_keep_all_text():
    sentences = [f"This is sentence number {i} of the document." for i in range(50)]
    text = " ".join(sentences)
    chunks = chunk_text(text, TOK, max_tokens=40)
    assert len(chunks) > 1
    assert all(count_tokens(c, TOK) <= 40 for c in chunks)
    assert " ".join(chunks) == text


def test_short_text_is_single_chunk():
    assert chunk_text("Just one short sentence.", TOK, max_tokens=900) == ["Just one short sentence."]


def test_oversized_sentence_gets_own_chunk():
    long_sentence = "word " * 100 + "end."
    chunks = chunk_text(f"Short one. {long_sentence} Another short.", TOK, max_tokens=20)
    assert any(count_tokens(c, TOK) > 20 for c in chunks)
    assert len(chunks) == 3


def test_rouge_identical_text_scores_one():
    df = rouge_table(["the cat sat on the mat"], ["the cat sat on the mat"])
    assert df.loc[0, "rouge1"] == 1.0 and df.loc[0, "rougeL"] == 1.0
