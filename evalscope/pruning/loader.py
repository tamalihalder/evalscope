"""Load per-sample scores from evalscope review JSONL files."""
import json
from pathlib import Path


def load_reference_scores(
    reviews_dir: str | Path,
    benchmark: str,
    score_key: str = "auto",
) -> dict[str, dict[int, float]]:
    """
    Parse all ``{benchmark}__{model}.jsonl`` files in *reviews_dir*.

    Returns
    -------
    {model_name: {sample_index: score_value}}

    score_key : "auto" detects "pass" (LiveCodeBench) or "acc" (AA-LCR / MMMU).
    """
    reviews_dir = Path(reviews_dir)
    result: dict[str, dict[int, float]] = {}

    for path in sorted(reviews_dir.glob(f"{benchmark}__*.jsonl")):
        model = path.stem.split("__", 1)[1]
        scores: dict[int, float] = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            obj = json.loads(line)
            idx = int(obj["index"])
            value = obj["sample_score"]["score"]["value"]
            if score_key == "auto":
                key = "pass" if "pass" in value else "acc"
            else:
                key = score_key
            scores[idx] = float(value[key])
        if scores:
            result[model] = scores

    return result
