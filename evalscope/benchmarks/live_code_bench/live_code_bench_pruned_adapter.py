"""
live_code_bench_pruned — pruning-aware alias for LiveCodeBench.

Registers the dataset name ``live_code_bench_pruned`` so that the command:

    evalscope eval --model <model> \\
        --datasets live_code_bench_pruned \\
        --dataset-args '{"pruning_strategy": "stratified_discriminative", "prune_ratio": 0.1}'

works out of the box.  Pruning is applied automatically by the hook in
``evalscope/pruning/hook.py`` which is called from
``DefaultDataAdapter.load_dataset()``.  This adapter is otherwise identical
to ``LiveCodeBenchAdapter``.
"""
from evalscope.api.benchmark import BenchmarkMeta
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags

from .live_code_bench_adapter import LiveCodeBenchAdapter


@register_benchmark(
    BenchmarkMeta(
        name='live_code_bench_pruned',
        pretty_name='Live-Code-Bench (Pruned)',
        tags=[Tags.CODING],
        description=(
            'Pruning-aware variant of LiveCodeBench. '
            'Pass ``pruning_strategy`` and ``prune_ratio`` via ``--dataset-args`` '
            'to run on a discriminative subset selected by the evalscope pruning '
            'extension (stratified_discriminative or mmmu_probe strategies).'
        ),
        dataset_id='evalscope/livecodebench_code_generation_lite_parquet',
        subset_list=[
            'release_latest',
            'release_v1', 'release_v2', 'release_v3',
            'release_v4', 'release_v5', 'release_v6',
            'v1', 'v1_v2', 'v1_v3', 'v1_v4', 'v1_v5', 'v1_v6',
            'v2', 'v2_v3', 'v2_v4', 'v2_v5', 'v2_v6',
            'v3', 'v3_v4', 'v3_v5', 'v3_v6',
            'v4', 'v4_v5', 'v4_v6',
            'v5', 'v5_v6',
            'v6',
        ],
        metric_list=['acc'],
        aggregation='mean_and_pass_at_k',
        eval_split='test',
        prompt_template=(
            '### Question:\n{question_content}\n\n'
            '{format_prompt} ### Answer: (use the provided format with backticks)\n\n'
        ),
        review_timeout=6,
        extra_params={
            'pruning_strategy': {
                'type': 'str',
                'description': (
                    'Pruning strategy to apply before evaluation. '
                    'Options: "stratified_discriminative" (LiveCodeBench/AA-LCR), '
                    '"mmmu_probe" (MMMU). '
                    'Omit to run the full benchmark without pruning.'
                ),
                'value': 'stratified_discriminative',
            },
            'prune_ratio': {
                'type': 'float',
                'description': 'Fraction of samples to keep (e.g. 0.1 keeps 10%).',
                'value': 0.1,
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
            'start_date': {
                'type': 'str | null',
                'description': 'Filter problems starting from this date (YYYY-MM-DD).',
                'value': None,
            },
            'end_date': {
                'type': 'str | null',
                'description': 'Filter problems up to this date (YYYY-MM-DD).',
                'value': None,
            },
            'debug': {
                'type': 'bool',
                'description': 'Enable verbose debug logging.',
                'value': False,
            },
        },
        sandbox_config={
            'image': 'python:3.11-slim',
            'tools_config': {
                'shell_executor': {},
                'python_executor': {},
            },
        },
    )
)
class LiveCodeBenchPrunedAdapter(LiveCodeBenchAdapter):
    """LiveCodeBench with automatic dataset pruning via evalscope/pruning/hook.py."""
