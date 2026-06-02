"""
evalscope-pruning
=================
Benchmark dataset pruning strategies for fast model quality evaluation.

Strategies
----------
stratified_discriminative : SDP — difficulty-stratified, informativeness-ranked
                            selection for LiveCodeBench and AA-LCR.
mmmu_probe                : Visual-dependency-weighted probe set for MMMU that
                            surfaces image-encoder degradation.

Quick usage
-----------
    from pruning import get_pruner, load_reference_scores

    scores = load_reference_scores("path/to/reviews", benchmark="aa_lcr")
    pruner = get_pruner("stratified_discriminative", prune_ratio=0.2)
    keep = pruner.select_indices(list(range(100)), ref_scores=scores)
"""

from .registry import get_pruner, list_strategies
from .loader import load_reference_scores

__all__ = ["get_pruner", "list_strategies", "load_reference_scores"]
