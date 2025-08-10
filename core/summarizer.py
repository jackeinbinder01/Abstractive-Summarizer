import time
from datasets import Dataset


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
        start = time.perf_counter()
        for i in range(n):
            document = corpus[i]
            summary = self.summarize(document["article"])
            documents.append({
                "article": document["article"],
                "highlights": document["highlights"],
                "summary": summary,
                "id": document["id"]
            })

        end = time.perf_counter()
        elapsed = end - start

        hours, remainder = divmod(elapsed, 3600)
        minutes, seconds = divmod(remainder, 60)

        print(f"Completed in {int(hours):02}:{int(minutes):02}:{seconds:.2f}")
        return documents


