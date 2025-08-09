from copy import deepcopy
from evaluation.rouge_evaluator import RougeEvaluator
import os


def sum_eval(corpus: list[dict], configs: dict) -> dict:
    scores = {}

    results_dir = os.path.join("..", "results")
    os.makedirs(results_dir, exist_ok=True)

    for label, summarizer in configs.items():
        print(f"Evaluating: {label.upper()}...\n")
        local_corpus = deepcopy(corpus)
        results = summarizer.structured_batch_summarize(local_corpus)

        # Defensive filtering of malformed entries
        valid = [
            doc for doc in results
            if isinstance(doc.get("summary"), str)
            and isinstance(doc.get("highlights"), str)
            and doc["summary"].strip() and doc["highlights"].strip()
        ]

        if not valid:
            print(f"[Warning] No valid summaries found for {label}. Skipping evaluation.")
            continue

        preds = [doc["summary"] for doc in valid]
        refs = [doc["highlights"] for doc in valid]

        evaluator = RougeEvaluator(rouge_types=['rouge1', 'rouge2', 'rougeL'])
        try:
            scores[label] = evaluator.evaluate(preds, refs)
            file_label = label.lower().replace(' ', '_')
            filename = f'{file_label}_re'
            save_path = os.path.join(results_dir, filename)
            evaluator.plot(label, save_path=save_path)
        except Exception as e:
            print(f"[Error] Evaluation failed for {label}: {e}")

    return scores
