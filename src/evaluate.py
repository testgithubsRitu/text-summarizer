"""Score summaries against human reference summaries with ROUGE.

ROUGE-1 / ROUGE-2 measure unigram / bigram overlap; ROUGE-L measures the longest common subsequence.
Run:  python -m src.evaluate data/eval_sample.csv
"""
import sys

import pandas as pd
from rouge_score import rouge_scorer


def rouge_table(predictions, references):
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    rows = []
    for pred, ref in zip(predictions, references):
        s = scorer.score(ref, pred)
        rows.append({k: round(v.fmeasure, 4) for k, v in s.items()})
    return pd.DataFrame(rows)


def main(path):
    from src.summarizer import Summarizer

    df = pd.read_csv(path)  # columns: text, reference
    summarizer = Summarizer()
    df["prediction"] = [summarizer.summarize(t) for t in df["text"]]
    scores = rouge_table(df["prediction"], df["reference"])
    print(scores.to_string())
    print("\nMean ROUGE F1:")
    print(scores.mean().round(4).to_string())


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/eval_sample.csv")
