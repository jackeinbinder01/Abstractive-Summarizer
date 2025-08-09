import random as r
import json

r.seed(42)


def sample(corpus: list[dict], k=1) -> dict | list[dict]:
    if k > len(corpus):
        raise ValueError(f"Requested {k} samples, but the corpus only has {len(corpus)}")
    indices = r.sample(range(len(corpus)), k=k)
    if k == 1:
        return corpus[indices[0]]
    return [corpus[i] for i in indices]


def print_sample(sample: dict, max_chars: int = None) -> None:
    for k, v in sample.items():
        print(f"{k.title()}:")
        try:
            v_str = json.dumps(v, indent=4, ensure_ascii=False)
        except TypeError:
            v_str = str(v)
        if max_chars and len(v_str) > max_chars:
            v_str = v_str[:max_chars] + "..."
        print(v_str)
        print()
