import matplotlib.pyplot as plt
import numpy as np

def plot_avg_rouge(model_scores: dict, save_path: str) -> None:

    models = list(model_scores.keys())
    rouge_types = ['ROUGE-1', 'ROUGE-2', 'ROUGE-L']

    scores = {
        'ROUGE-1': [model_scores[m]['rouge1'] for m in models],
        'ROUGE-2': [model_scores[m]['rouge2'] for m in models],
        'ROUGE-L': [model_scores[m]['rougeL'] for m in models],
    }

    x = np.arange(len(models))
    width = 0.2

    fig, ax = plt.subplots(figsize=(16, 6))

    for i, rouge_type in enumerate(rouge_types):
        offset = (i - 1) * width
        bar_positions = x + offset
        bar_heights = scores[rouge_type]
        bars = ax.bar(bar_positions, bar_heights, width, label=rouge_type)

        for xpos, height in zip(bar_positions, bar_heights):
            ax.text(xpos, height + 0.015, f"{height:.2f}", ha='center', va='bottom', fontsize=9)

    max_score = max(max(s) for s in scores.values())
    ax.set_ylim(0, min(1.0, max_score + 0.3))
    ax.set_ylabel("ROUGE Score", labelpad=10)
    ax.set_title("Average ROUGE Scores by Model", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace('_', ' ').upper() for m in models], rotation=20, ha='right')
    ax.legend(title="ROUGE Metric")

    plt.tight_layout()
    plt.grid(False)
    if save_path:
        plt.savefig(f"{save_path}_re.png")
    plt.show()

def plot_avg_manual_score(ablation_avg_manual_scores: dict, save_path: str) -> None:
    models = list(ablation_avg_manual_scores.keys())
    scores = list(ablation_avg_manual_scores.values())

    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(models, scores, color='skyblue')

    ax.set_ylim(1, 6)
    ax.set_ylabel("Manual Score (1–5)", labelpad=10)
    ax.set_title("Average Manual Scores by Model", pad=15)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([m.replace('_', ' ').upper() for m in models], rotation=20, ha='right')

    # Add bar height labels
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.1,
            f"{height:.2f}",
            ha='center', va='bottom'
        )

    plt.suptitle("AVERAGE MANUAL SCORE", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.grid(False)

    if save_path:
        plt.savefig(f"{save_path}_me.png")
    plt.show()