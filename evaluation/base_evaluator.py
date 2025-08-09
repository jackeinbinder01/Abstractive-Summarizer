from abc import ABC, abstractmethod


class BaseEvaluator(ABC):

    @abstractmethod
    def evaluate(self, predictions: list[str], references: list[str]) -> dict[str, float]:
        """Compute evaluation scores."""
        pass

    @abstractmethod
    def print_report(self, label: str) -> None:
        """Print evaluation report."""
        pass

    @abstractmethod
    def plot(self, metric: str) -> None:
        """Plot evaluation scores."""
        pass
