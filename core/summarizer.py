import time
from datasets import Dataset


class Summarizer:
    def __init__(self, model, **gen_defaults):
        self.model = model
        self.gen_defaults = gen_defaults

    def summarize(self, article: str, **kwargs) -> str:
        params = {**self.gen_defaults, **kwargs}
        out = self.model(article, **params)[0]["summary_text"]
        return out

    def batch_summarize(self, corpus: list[str], **kwargs) -> list[str]:
        return [self.summarize(doc, **kwargs) for doc in corpus]

    def structured_batch_summarize(self, corpus: Dataset, max_articles=None, **kwargs) -> list[dict]:
        documents = []
        n = len(corpus) if max_articles is None else min(max_articles, len(corpus))
        if n == 0:
            print("Corpus is empty.")
            return []

        print(f"Generating summaries for {n} articles...\n")
        start = time.perf_counter()
        for i in range(n):
            document = corpus[i]
            summary = self.summarize(document["article"], **kwargs)
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


