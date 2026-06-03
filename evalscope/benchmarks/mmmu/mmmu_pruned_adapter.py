"""
mmmu_pruned — pruning-aware alias for MMMU.

Registers the dataset name ``mmmu_pruned`` so that the command:

    evalscope eval --model <model> \\
        --datasets mmmu_pruned \\
        --dataset-args '{"pruning_strategy": "mmmu_probe", "prune_ratio": 0.05}'

works out of the box.  Pruning is applied automatically by the hook in
``evalscope/pruning/hook.py`` which is called from
``DefaultDataAdapter.load_dataset()``.  This adapter is otherwise identical
to ``MMMUAdapter``.

The hook detects ``"mmmu"`` in the benchmark name and automatically extracts
per-sample visual metadata (subject, n_images, question_tokens) to feed the
MMUProbePruner, which weights subjects by visual dependency — prioritising
items where the image carries irreducible information (Electronics, Chemistry,
Diagnostics, etc.) over text-heavy subjects (History, Literature, Psychology).
"""
from evalscope.api.benchmark import BenchmarkMeta
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags

from .mmmu_adapter import MMMUAdapter, SUBSET_LIST, OPEN_PROMPT


@register_benchmark(
    BenchmarkMeta(
        name='mmmu_pruned',
        pretty_name='MMMU (Pruned)',
        tags=[Tags.MULTI_MODAL, Tags.KNOWLEDGE, Tags.QA],
        description=(
            'Pruning-aware variant of MMMU. '
            'Pass ``pruning_strategy`` and ``prune_ratio`` via ``--dataset-args`` '
            'to run on a subset selected by the evalscope pruning extension. '
            'The mmmu_probe strategy weights subjects by visual dependency, '
            'making the pruned set a targeted probe for image-encoder degradation '
            'rather than a random sample of the full benchmark.'
        ),
        dataset_id='AI-ModelScope/MMMU',
        subset_list=SUBSET_LIST,
        metric_list=['acc'],
        eval_split='validation',
        prompt_template=OPEN_PROMPT,
        extra_params={
            'pruning_strategy': {
                'type': 'str',
                'description': (
                    'Pruning strategy to apply before evaluation. '
                    'Options: "mmmu_probe" (default, image-encoder degradation probe), '
                    '"stratified_discriminative" (generic difficulty stratification). '
                    'Omit to run the full benchmark without pruning.'
                ),
                'value': 'mmmu_probe',
            },
            'prune_ratio': {
                'type': 'float',
                'description': 'Fraction of MMMU to keep (e.g. 0.05 keeps ~600/12K samples).',
                'value': 0.05,
            },
            'reference_dir': {
                'type': 'str | null',
                'description': (
                    'Path to directory containing '
                    '``{benchmark}__{model}.jsonl`` reference score files. '
                    'When provided, items where some models fail (but not all) '
                    'receive an additional boost — these are likely encoder failures '
                    'rather than universally hard questions.'
                ),
                'value': None,
            },
            'min_per_subject': {
                'type': 'int',
                'description': 'Minimum samples to keep per high-visual-dependency subject.',
                'value': 3,
            },
            'debug': {
                'type': 'bool',
                'description': 'Enable verbose debug logging.',
                'value': False,
            },
        },
    )
)
class MMMUPrunedAdapter(MMMUAdapter):
    """MMMU with automatic dataset pruning via evalscope/pruning/hook.py."""
