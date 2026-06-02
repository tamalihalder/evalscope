"""
evalscope integration hook
===========================
This module is installed alongside evalscope (in evalscope/pruning/) and
provides the _apply_pruning() method that DefaultDataAdapter calls at the
end of load_dataset().

To integrate into a forked evalscope, apply the patch in
  evalscope_integration/default_data_adapter.patch
which adds two things to DefaultDataAdapter:

1. Import this module at the top of default_data_adapter.py
2. Call self._apply_pruning() at the end of load_dataset()
"""
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def apply_pruning(
    test_dataset,
    benchmark_name: str,
    extra_params: dict,
) -> Any:
    """
    Called from DefaultDataAdapter.load_dataset() when pruning params are present.

    Parameters
    ----------
    test_dataset   : the DatasetDict / list returned by load_dataset()
    benchmark_name : e.g. "live_code_bench_v5", "aa_lcr", "mmmu"
    extra_params   : full dict from --dataset-args
    """
    strategy = extra_params.get("pruning_strategy")
    if not strategy:
        return test_dataset

    prune_ratio = float(extra_params.get("prune_ratio", 0.2))

    # Resolve reference scores directory
    ref_dir = extra_params.get("reference_dir")
    ref_scores = None
    if ref_dir:
        from pruning.loader import load_reference_scores
        ref_dir = Path(ref_dir)
        if ref_dir.exists():
            ref_scores = load_reference_scores(ref_dir, benchmark=benchmark_name)
            logger.info(
                f"[Pruning] Loaded reference scores from {ref_dir} "
                f"for {len(ref_scores)} model(s)."
            )
        else:
            logger.warning(f"[Pruning] reference_dir not found: {ref_dir}. Proceeding without reference scores.")

    # Extract all indices from the dataset
    # evalscope DatasetDict maps split -> list[Sample]; we work on the test split
    if isinstance(test_dataset, dict):
        split_key = "test" if "test" in test_dataset else next(iter(test_dataset))
        samples = test_dataset[split_key]
    else:
        samples = test_dataset
        split_key = None

    all_indices = [getattr(s, "index", i) for i, s in enumerate(samples)]

    # Build strategy-specific extra context
    pruner_kwargs: dict = {}

    # For AA-LCR: pass token counts for context-length stratification
    if "lcr" in benchmark_name.lower() or "lcr" in strategy:
        token_counts = {}
        for s in samples:
            meta = getattr(s, "metadata", {}) or {}
            idx = getattr(s, "index", None)
            if idx is not None and "input_tokens" in meta:
                token_counts[idx] = int(meta["input_tokens"])
        if token_counts:
            extra_params = {**extra_params, "_token_counts": token_counts}

    # For MMMU probe: extract per-sample visual metadata
    if "mmmu" in benchmark_name.lower():
        from pruning.mmmu_probe import MMUProbePruner
        sample_meta = {}
        for s in samples:
            idx = getattr(s, "index", None)
            if idx is not None:
                sample_meta[idx] = MMUProbePruner.infer_meta_from_sample(s)
        if sample_meta:
            extra_params = {**extra_params, "_sample_meta": sample_meta}

    # Get the pruner and select indices
    from pruning.registry import get_pruner
    pruner = get_pruner(strategy, prune_ratio=prune_ratio, **pruner_kwargs)
    keep_indices = pruner.select_indices(
        all_indices=all_indices,
        ref_scores=ref_scores,
        extra_params=extra_params,
    )

    # Filter the dataset
    pruned_samples = [s for s, idx in zip(samples, all_indices) if idx in keep_indices]

    n_before = len(samples)
    n_after = len(pruned_samples)
    logger.info(
        f"[Pruning] strategy={strategy}, ratio={prune_ratio} — "
        f"kept {n_after}/{n_before} samples ({n_after/n_before*100:.1f}%)"
    )

    if split_key is not None:
        return {**test_dataset, split_key: pruned_samples}
    return pruned_samples
