"""
MMMU Image-Encoder Degradation Probe
======================================
Selects a probe set from the full 12K MMMU dataset that specifically surfaces
image-encoder degradation, as opposed to generic language-model capability gaps.

Design rationale
----------------
A degraded image encoder causes failures on questions whose answer requires
*visual understanding* that cannot be recovered from the text of the question
alone.  We therefore prioritise subjects where images carry irreducible
information:

  HIGH visual dependency
    Electronics          (circuit diagrams, schematics)
    Chemistry            (molecular structures, reaction diagrams)
    Diagnostics / Lab    (microscopy, histology, lab readouts)
    Clinical Medicine    (X-ray, MRI, ultrasound)
    Computer Science     (algorithm diagrams, UML)
    Art / Design         (visual composition, style analysis)
    Architecture         (blueprints, spatial layouts)
    Geography            (maps, satellite imagery)
    Math                 (geometric figures, graphs)
    Biology              (cell diagrams, anatomy)

  LOW visual dependency (excluded or down-weighted)
    History, Literature, Economics, Finance, Psychology
    (these can often be answered from the question text or world knowledge)

Within high-dependency subjects the strategy further filters by:
  1. Number of images >= 2   (multi-image questions rely more on visual reasoning)
  2. Question tokens < 200   (shorter questions reduce text-only shortcutting)

When reference scores are available, items are additionally sorted by:
  - questions where the model scored low (potential encoder failure signal)
  - while excluding items where ALL reference models scored low
    (those are probably just hard questions, not encoder failures)

Usage
-----
evalscope eval --model <name> --datasets mmmu \
    --dataset-args '{"pruning_strategy": "mmmu_probe", "prune_ratio": 0.05}'
"""
import math
import re
from .base import BasePruner

# Subjects and their visual-dependency weight (0 = skip, 1 = low, 2 = high)
VISUAL_DEPENDENCY: dict[str, int] = {
    "Electronics": 2,
    "Chemistry": 2,
    "Diagnostics_and_Laboratory_Medicine": 2,
    "Clinical_Medicine": 2,
    "Computer_Science": 2,
    "Art": 2,
    "Art_Theory": 2,
    "Architecture_and_Engineering": 2,
    "Geography": 2,
    "Biology": 2,
    "Math": 2,
    "Physics": 2,
    "Energy_and_Power": 2,
    "Materials": 2,
    "Mechanical_Engineering": 2,
    "Basic_Medical_Science": 2,
    "Agriculture": 1,
    "Accounting": 1,
    "Economics": 1,
    "Finance": 1,
    "Manage": 1,
    "Marketing": 1,
    "History": 0,
    "Literature": 0,
    "Psychology": 0,
    "Sociology": 0,
    "Music": 0,
}


class MMUProbePruner(BasePruner):
    """
    Parameters (via dataset-args)
    --------------------------------
    prune_ratio           : float  fraction of MMMU to keep (e.g. 0.05 = 600/12K)
    min_per_subject       : int    minimum kept samples per high-dep subject (default 3)
    require_multi_image   : bool   prefer items with >=2 images (default False)
    """

    def __init__(
        self,
        prune_ratio: float,
        min_per_subject: int = 3,
        require_multi_image: bool = False,
        **kwargs,
    ):
        super().__init__(prune_ratio)
        self.min_per_subject = min_per_subject
        self.require_multi_image = require_multi_image

    def select_indices(
        self,
        all_indices: list[int],
        ref_scores: dict[str, dict[int, float]] | None = None,
        extra_params: dict | None = None,
    ) -> set[int]:
        extra_params = extra_params or {}
        target_n = max(1, math.ceil(self.prune_ratio * len(all_indices)))

        # Sample metadata: {index: {subject, n_images, question_tokens}}
        sample_meta: dict[int, dict] = extra_params.get("_sample_meta", {})

        # ---- score each index ----
        scores: dict[int, float] = {}
        for idx in all_indices:
            meta = sample_meta.get(idx, {})
            subject = meta.get("subject", "")
            dep = VISUAL_DEPENDENCY.get(subject, 1)
            if dep == 0:
                continue  # skip text-heavy subjects

            # Base score: visual dependency weight
            score = float(dep)

            # Boost for multi-image items
            if meta.get("n_images", 0) >= 2:
                score += 1.0

            # Boost for shorter questions (less text-only shortcutting risk)
            q_tokens = meta.get("question_tokens", 999)
            if q_tokens < 100:
                score += 0.5

            # Boost for items that discriminate based on reference scores
            if ref_scores:
                model_scores = [
                    ref_scores[m][idx]
                    for m in ref_scores
                    if idx in ref_scores[m]
                ]
                if model_scores:
                    mean_score = sum(model_scores) / len(model_scores)
                    # Prefer items where some models fail (potential encoder signal)
                    # but not universally hard items (all fail)
                    if 0 < mean_score < 0.9:
                        score += (1.0 - mean_score) * 2.0

            scores[idx] = score

        # ---- stratify by subject, allocate budget proportionally ----
        subjects: dict[str, list[int]] = {}
        for idx, s in scores.items():
            sub = sample_meta.get(idx, {}).get("subject", "__unknown__")
            subjects.setdefault(sub, []).append(idx)

        # Weight per subject = sum of item scores
        sub_weights = {
            sub: sum(scores[i] for i in idxs)
            for sub, idxs in subjects.items()
        }
        total_w = sum(sub_weights.values()) or 1.0

        keep: set[int] = set()
        for sub, idxs in subjects.items():
            quota = max(
                self.min_per_subject,
                math.floor(sub_weights[sub] / total_w * target_n),
            )
            quota = min(quota, len(idxs))
            ranked = sorted(idxs, key=lambda i: -scores[i])
            keep.update(ranked[:quota])

        # Trim to target if over-allocated
        if len(keep) > target_n:
            keep = set(sorted(keep, key=lambda i: -scores[i])[:target_n])

        return keep

    @staticmethod
    def infer_meta_from_sample(sample) -> dict:
        """
        Extract {subject, n_images, question_tokens} from an evalscope Sample.
        Called during dataset loading when _sample_meta is not pre-computed.
        """
        question = getattr(sample, "query", "") or ""
        n_images = len(re.findall(r"<image>|<img|!\[", question, re.IGNORECASE))
        return {
            "subject": getattr(sample, "subset_name", ""),
            "n_images": n_images,
            "question_tokens": len(question.split()),
        }
