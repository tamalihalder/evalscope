from abc import ABC, abstractmethod


class BasePruner(ABC):
    """Select a representative subset of benchmark samples before evaluation."""

    def __init__(self, prune_ratio: float, **kwargs):
        if not 0 < prune_ratio <= 1:
            raise ValueError(f"prune_ratio must be in (0, 1], got {prune_ratio}")
        self.prune_ratio = prune_ratio

    @abstractmethod
    def select_indices(
        self,
        all_indices: list[int],
        ref_scores: dict[str, dict[int, float]] | None = None,
        extra_params: dict | None = None,
    ) -> set[int]:
        """
        Return the set of sample indices to keep.

        Parameters
        ----------
        all_indices   : ordered list of all sample indices in the dataset
        ref_scores    : optional {model_name: {index: score}} from prior runs
        extra_params  : raw dataset-args dict for strategy-specific config
        """
