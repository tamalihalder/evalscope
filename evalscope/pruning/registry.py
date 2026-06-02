from .stratified_discriminative import StratifiedDiscriminativePruner
from .mmmu_probe import MMUProbePruner
from .base import BasePruner

_REGISTRY: dict[str, type[BasePruner]] = {
    "stratified_discriminative": StratifiedDiscriminativePruner,
    "mmmu_probe": MMUProbePruner,
}


def get_pruner(strategy: str, prune_ratio: float, **kwargs) -> BasePruner:
    cls = _REGISTRY.get(strategy)
    if cls is None:
        raise ValueError(
            f"Unknown pruning strategy '{strategy}'. "
            f"Available: {sorted(_REGISTRY)}"
        )
    return cls(prune_ratio=prune_ratio, **kwargs)


def list_strategies() -> list[str]:
    return sorted(_REGISTRY)
