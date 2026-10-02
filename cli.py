"""python cli.py path/to/article.txt [--max-length 130]"""
import argparse
from pathlib import Path

from src.summarizer import Summarizer


def main():
    parser = argparse.ArgumentParser(description="Summarize a text file")
    parser.add_argument("file")
    parser.add_argument("--max-length", type=int, default=130, help="max tokens in summary")
    parser.add_argument("--min-length", type=int, default=30, help="min tokens in summary")
    args = parser.parse_args()
    text = Path(args.file).read_text(encoding="utf-8")
    print(Summarizer().summarize(text, args.max_length, args.min_length))


if __name__ == "__main__":
    main()
