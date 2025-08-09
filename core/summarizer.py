from datasets import Dataset
from summarizer.utils.timer import Timer
from ..preprocessing.base_text_preprocessor import BaseTextPreprocessor

class Summarizer:
    def __init__(self, model):
        self.model = model

    def summarize(self, article: str, min_length=30, max_length=100, **kwargs) -> str:
        summary = self.model(
            article,
            max_length=max_length,
            min_length=min_length,
            **kwargs
        )[0]["summary_text"]
        return summary

    def batch_summarize(self, corpus: list[str]) -> list[str]:
        return [self.summarize(doc) for doc in corpus]

    def structured_batch_summarize(self, corpus: Dataset, max_articles=None) -> list[dict]:
        documents = []
        n = len(corpus) if max_articles is None else min(max_articles, len(corpus))
        if n == 0:
            print("Corpus is empty.")
            return []

        print(f"Generating summaries for {n} articles...\n")
        timer = Timer()
        timer.start()
        for i in range(n):
            document = corpus[i]
            summary = self.summarize(document["article"])
            documents.append({
                "article": document["article"],
                "highlights": document["highlights"],
                "summary": summary,
                "id": document["id"]
            })

        timer.stop()
        timer.print("Completed in")
        return documents
