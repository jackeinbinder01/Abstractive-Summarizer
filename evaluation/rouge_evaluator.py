from evaluation.base_evaluator import BaseEvaluator
from rouge_score import rouge_scorer
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


class RougeEvaluator(BaseEvaluator):
    VALID_ROUGE = {'rouge1', 'rouge2', 'rougeL'}

    def __init__(
            self,
            rouge_types: list[str] = None,
            use_stemmer: bool = True,
            split_summaries: bool = False,
            tokenizer=None
    ):
        # Public
        self.rouge_types = rouge_types or ['rouge1', 'rouge2', 'rougeL']

        invalid_metrics = [m for m in self.rouge_types if m not in self.VALID_ROUGE]
        if invalid_metrics:
            raise ValueError(f"Invalid ROUGE types: {invalid_metrics}")
        self.use_stemmer = use_stemmer
        self.split_summaries = split_summaries
        self.tokenizer = tokenizer

        # Private
        self._scorer = rouge_scorer.RougeScorer(
            self.rouge_types,
            use_stemmer=self.use_stemmer,
            split_summaries=self.split_summaries,
            tokenizer=self.tokenizer
        )
        self._scores: list[dict[str, float]] = []
        self._aggregate_scores: dict[str, float] = {}

    def _check_scores(self) -> bool:
        if not self._scores:
            print("No scores available. Run evaluate() first.")
            return False
        return True

    @property
    def scores(self) -> list[dict[str, float]]:
        return self._scores.copy()

    @property
    def aggregate(self) -> dict[str, float]:
        return self._aggregate_scores.copy()

    def evaluate(self, predictions: list[str], references: list[str]) -> dict[str, float]:
        if len(predictions) != len(references):
            raise ValueError("Predictions and references must be the same length.")

        self._scores.clear()
        aggregates = {metric: [] for metric in self.rouge_types}

        for pred, ref in zip(predictions, references):
            result = self._scorer.score(ref, pred)
            row = {}
            for metric in self.rouge_types:
                f1 = result[metric].fmeasure
                row[metric] = f1
                aggregates[metric].append(f1)
            self._scores.append(row)
        self._aggregate_scores = {
            metric: sum(values) / len(values) if values else 0.0
            for metric, values in aggregates.items()
        }

        return self._aggregate_scores

    def print_report(self, label: str, metrics: list[str] = None) -> None:
        if not self._check_scores():
            return

        metrics = metrics or self.rouge_types
        print(f"ROUGE Evaluation Report - {label.upper()}")
        for metric in metrics:
            score = self._aggregate_scores.get(metric)
            if score is not None:
                print(f"{metric.upper():>7}: {score:.4f}")
            else:
                print(f"{metric.upper():>7}: Not supported")

    def plot(self, label: str, metrics: list[str] = None, save_path: str = None) -> None:
        if not self._check_scores():
            return

        metrics = metrics or self.rouge_types
        valid_metrics = [m for m in metrics if m in self.rouge_types]
        if not valid_metrics:
            print("No valid metrics to plot.")
            return

        fig = plt.figure(figsize=(14, 8))
        gs = GridSpec(2, 2, figure=fig)

        metric_axes = {
            'rouge1': fig.add_subplot(gs[0, 0]),
            'rouge2': fig.add_subplot(gs[0, 1]),
            'rougeL': fig.add_subplot(gs[1, :])
        }

        for metric in ['rouge1', 'rouge2', 'rougeL']:
            if metric not in valid_metrics:
                continue

            ax = metric_axes[metric]
            values = [row[metric] for row in self._scores]
            avg = sum(values) / len(values) if values else 0.0

            ax.plot(values, marker='o', linestyle='-', alpha=0.75, label='F1 Score by Article')
            ax.axhline(avg, color='gray', linestyle='--', linewidth=1, label=f"Avg F1 Score: {avg:.4f}")

            ax.set_title(f"{label.upper().replace('_', ' ')} - {metric.upper()}", pad=15)
            ax.set_xlabel("Example Index", labelpad=10)
            ax.set_ylabel("F1 Score", labelpad=10)
            ax.set_ylim(0, 1)
            ax.legend(loc='upper right')
            ax.grid(False)

        plt.suptitle(f"{label.upper().replace('_', ' ')} - ROUGE METRICS", fontsize=16, y=1.02)
        plt.tight_layout()

        if save_path:
            plt.savefig(f"{save_path}.png")
        plt.show()

