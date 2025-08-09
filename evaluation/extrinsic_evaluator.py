from evaluation.manual_score import ManualScore
import matplotlib.pyplot as plt
from collections import Counter


class ExtrinsicEvaluator:
    def __init__(self, articles: list[dict] = None):
        if not articles:
            raise ValueError(f"Articles must be provided.")
        self.articles = articles
        self.scores: dict[str, ManualScore] = {}
        self.avg_score = 0.0

    def evaluate(self, label):
        print(f"Manually evaluating {label.upper()}...")

        for idx, article in enumerate(self.articles):
            article_id = str(article.get('id', idx))
            print(f"ID: {article_id}")
            print(f"Article:\n{article['article']}\n")
            print(f"Highlights:\n{article['highlights']}\n")
            print(f"Summary:\n{article['summary']}\n")

            print(f"How would you rate the summary for {article_id}?\n")
            for score in ManualScore:
                print(f"  {score.value}. {score.name.replace('_', ' ').title()}")
            print()

    def submit_scores(self, id_to_score: dict[str, int]) -> None:
        for idx, article in enumerate(self.articles):
            article_id = str(article.get('id', idx))
            score_value = id_to_score.get(article_id)

            if score_value in [s.value for s in ManualScore]:
                self.scores[article_id] = ManualScore(score_value)
            else:
                print(f"Invalid or missing score for ID {article_id}. Skipping.")

        values = [score.value for score in self.scores.values()]
        avg = sum(values) / len(values) if values else 0.0
        self.avg_score = avg

    def plot(self, label, save_path: str = None) -> None:
        values = [score.value for score in self.scores.values()]

        categories = [s.value for s in ManualScore]
        labels = [f"{s.name.replace('_', ' ').title()} - {s.value}" for s in ManualScore]
        counts = Counter(values)
        freqs = [counts.get(val, 0) for val in categories]

        fig, axs = plt.subplots(1, 2, figsize=(12, 4))

        # Left bar chart
        axs[0].bar(labels, freqs, color='skyblue')
        axs[0].set_title(f"{label.upper().replace('_', ' ')} - Manual Score Distribution")
        axs[0].set_ylabel("Number of Articles", labelpad=10)
        axs[0].set_xticks(range(len(labels)))
        axs[0].set_xticklabels(labels, rotation=15)
        axs[0].set_ylim(0, max(freqs) + 1)

        # Right bar chart
        bar_avg = axs[1].bar(["Average Score"], [self.avg_score], color='skyblue')
        axs[1].set_ylim(1, 6)
        axs[1].set_title(f"{label.upper().replace('_', ' ')} - Average Manual Score", pad=10)
        axs[1].set_ylabel("Manual Score (1-5)", labelpad=10)

        for bar in bar_avg:
            height = bar.get_height()
            axs[1].text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.1,
                f"{height:.2f}",
                ha='center', va='bottom'
            )

        plt.suptitle(f"{label.upper().replace('_', ' ')} - Manual Evaluation", fontsize=16, y=1.02)
        plt.grid(False)
        plt.tight_layout()

        if save_path:
            plt.savefig(f"{save_path}_me.png")
        plt.show()

    def print_report(self, label: str) -> None:
        print(f"{label}")
        for idx, article in enumerate(self.articles):
            article_id = str(article.get('id', idx))
            print(f"ID:\n{article_id}")
            print(f"Article:\n{article['article']}")
            print(f"Highlights:\n{article['highlights']}")
            print(f"Summary:\n{article['summary']}")
