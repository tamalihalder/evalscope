"""
Stratified Discriminative Pruning (SDP)
========================================
Selects the smallest sample set that preserves the quality signal by:

1. Computing per-sample difficulty  p_i = mean(scores_i across models)
   and IRT informativeness          I_i = p_i * (1 - p_i)

2. Partitioning samples into difficulty strata:
     easy   : p > 2/3   (most models pass)
     medium : 1/3 <= p <= 2/3
     hard   : p < 1/3   (most models fail)

3. Allocating budget proportionally to stratum size × mean informativeness,
   with a minimum floor of 1 per non-empty stratum.

4. Within each stratum, selecting samples in descending informativeness order
   (ties broken by index for reproducibility).

For AA-LCR, an optional second axis stratifies by context length
(short / long, split at the median input_tokens).

Rationale
---------
- "All-pass" and "all-fail" items have I=0 and receive the lowest priority,
  so they are dropped first when the budget is tight.
- Stratified allocation ensures difficulty coverage across the full range,
  preventing collapse to a single difficulty tier.
- No model names, ranks, or identities are encoded in the selection — the
  algorithm generalises to any unseen model.
"""
import math
from .base import BasePruner


class StratifiedDiscriminativePruner(BasePruner):
    """
    Parameters (via dataset-args)
    --------------------------------
    prune_ratio       : float  fraction of samples to KEEP  (e.g. 0.1 keeps 10%)
    n_strata          : int    number of difficulty bins     (default 3)
    min_per_stratum   : int    minimum samples per non-empty bin (default 1)
    token_strata      : bool   also stratify AA-LCR by input_tokens (default True)
    """

    def __init__(
        self,
        prune_ratio: float,
        n_strata: int = 3,
        min_per_stratum: int = 1,
        token_strata: bool = True,
        **kwargs,
    ):
        super().__init__(prune_ratio)
        self.n_strata = n_strata
        self.min_per_stratum = min_per_stratum
        self.token_strata = token_strata

    # ------------------------------------------------------------------
    def select_indices(
        self,
        all_indices: list[int],
        ref_scores: dict[str, dict[int, float]] | None = None,
        extra_params: dict | None = None,
    ) -> set[int]:
        if not ref_scores:
            # Fallback: keep a proportional sample using deterministic stride
            return self._stride_sample(all_indices)

        models = list(ref_scores.keys())
        target_n = max(1, math.ceil(self.prune_ratio * len(all_indices)))

        # ---- per-sample statistics ----
        stats: dict[int, dict] = {}
        for idx in all_indices:
            model_scores = [ref_scores[m][idx] for m in models if idx in ref_scores[m]]
            if not model_scores:
                continue
            p = sum(model_scores) / len(model_scores)
            info = p * (1.0 - p)
            stats[idx] = {"p": p, "info": info}

        # ---- optionally add token-length axis ----
        token_counts: dict[int, int] = (extra_params or {}).get("_token_counts", {})
        use_tokens = self.token_strata and bool(token_counts)
        if use_tokens:
            median_tokens = sorted(token_counts.values())[len(token_counts) // 2]

        # ---- assign strata ----
        def stratum_key(idx: int) -> tuple:
            p = stats[idx]["p"]
            diff_bin = self._difficulty_bin(p)
            if use_tokens:
                tok_bin = 0 if token_counts.get(idx, 0) <= median_tokens else 1
                return (diff_bin, tok_bin)
            return (diff_bin,)

        strata: dict[tuple, list[int]] = {}
        for idx in stats:
            key = stratum_key(idx)
            strata.setdefault(key, []).append(idx)

        # ---- allocate budget across strata ----
        keep: set[int] = set()
        quotas = self._allocate_budget(strata, stats, target_n)

        for key, quota in quotas.items():
            candidates = sorted(
                strata[key],
                key=lambda i: (-stats[i]["info"], i),  # high info first, then low index
            )
            keep.update(candidates[:quota])

        return keep

    # ------------------------------------------------------------------
    def _difficulty_bin(self, p: float) -> int:
        """0=easy, 1=medium, 2=hard"""
        boundaries = [i / self.n_strata for i in range(1, self.n_strata)]
        for i, boundary in enumerate(boundaries):
            if (1.0 - p) < boundary:   # (1-p) = difficulty
                return i
        return self.n_strata - 1

    def _allocate_budget(
        self,
        strata: dict[tuple, list[int]],
        stats: dict[int, dict],
        target_n: int,
    ) -> dict[tuple, int]:
        """
        Allocate target_n slots across strata.
        Weight = stratum_size * mean_informativeness + epsilon.
        Minimum self.min_per_stratum per non-empty stratum.
        """
        EPS = 1e-9
        weights: dict[tuple, float] = {}
        for key, members in strata.items():
            mean_info = sum(stats[i]["info"] for i in members) / len(members)
            weights[key] = len(members) * mean_info + EPS

        total_w = sum(weights.values())
        quotas: dict[tuple, int] = {}

        # First pass: proportional allocation
        remaining = target_n
        for key in strata:
            q = max(self.min_per_stratum, math.floor(weights[key] / total_w * target_n))
            q = min(q, len(strata[key]))  # can't take more than available
            quotas[key] = q
            remaining -= q

        # Second pass: distribute leftover to strata with most room
        if remaining > 0:
            order = sorted(strata.keys(), key=lambda k: -weights[k])
            for key in order:
                if remaining <= 0:
                    break
                headroom = len(strata[key]) - quotas[key]
                add = min(headroom, remaining)
                quotas[key] += add
                remaining -= add

        return quotas

    def _stride_sample(self, all_indices: list[int]) -> set[int]:
        """Deterministic evenly-spaced selection when no reference scores exist."""
        target_n = max(1, math.ceil(self.prune_ratio * len(all_indices)))
        step = len(all_indices) / target_n
        return {all_indices[round(i * step)] for i in range(target_n)}
