"""
aa_lcr_pruned — pruning-aware alias for AA-LCR.

Registers the dataset name ``aa_lcr_pruned`` so that the command:

    evalscope eval --model <model> \\
        --datasets aa_lcr_pruned \\
        --dataset-args '{"pruning_strategy": "stratified_discriminative", "prune_ratio": 0.2}'

works out of the box.  Pruning is applied automatically by the hook in
``evalscope/pruning/hook.py`` which is called from
``DefaultDataAdapter.load_dataset()``.  This adapter is otherwise identical
to ``AALCRAdapter``.

The hook detects ``"lcr"`` in the benchmark name and automatically enables
context-length stratification (token_strata=True), so short and long documents
are proportionally represented in the pruned set.
"""
from evalscope.api.benchmark import BenchmarkMeta
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags

from .aa_lcr_adapter import (
    AALCRAdapter,
    PROMPT_TEMPLATE,
    DOWNLOAD_URL,
    DEFAULT_CACHE_SUBDIR,
)


@register_benchmark(
    BenchmarkMeta(
        name='aa_lcr_pruned',
        pretty_name='AA-LCR (Pruned)',
        tags=[Tags.KNOWLEDGE, Tags.REASONING, Tags.LONG_CONTEXT],
        description=(
            'Pruning-aware variant of AA-LCR. '
            'Pass ``pruning_strategy`` and ``prune_ratio`` via ``--dataset-args`` '
            'to run on a discriminative subset selected by the evalscope pruning '
            'extension. The stratified_discriminative strategy additionally '
            'stratifies by context length (token_strata) to preserve coverage '
            'across short and long documents.'
        ),
        dataset_id='evalscope/AA-LCR',
        metric_list=['acc'],
        few_shot_num=0,
        train_split=None,
        eval_split='test',
        prompt_template=PROMPT_TEMPLATE,
        extra_params={
            'pruning_strategy': {
                'type': 'str',
                'description': (
                    'Pruning strategy to apply before evaluation. '
                    'Options: "stratified_discriminative" (default for AA-LCR). '
                    'Omit to run the full benchmark without pruning.'
                ),
                'value': 'stratified_discriminative',
            },
            'prune_ratio': {
                'type': 'float',
                'description': 'Fraction of samples to keep (e.g. 0.2 keeps 20%).',
                'value': 0.2,
            },
            'reference_dir': {
                'type': 'str | null',
                'description': (
                    'Path to directory containing '
                    '``{benchmark}__{model}.jsonl`` reference score files. '
                    'Required for informativeness-based pruning; '
                    'falls back to deterministic stride sampling when absent.'
                ),
                'value': None,
            },
            'text_dir': {
                'type': 'str | null',
                'description': (
                    'Local directory containing extracted AA-LCR text files; '
                    'if null will auto-download & extract.'
                ),
                'value': None,
            },
            'debug': {
                'type': 'bool',
                'description': 'Enable verbose debug logging.',
                'value': False,
            },
        },
    )
)
class AALCRPrunedAdapter(AALCRAdapter):
    """AA-LCR with automatic dataset pruning via evalscope/pruning/hook.py."""
