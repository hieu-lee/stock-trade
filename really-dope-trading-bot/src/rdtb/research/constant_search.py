from __future__ import annotations

from rdtb.research.auto_search import (
    AutoSearchReport as ConstantSearchReport,
    auto_search_to_dict as constant_search_to_dict,
    render_auto_search_markdown as render_constant_search_markdown,
    run_auto_search,
)
from rdtb.research.search_scoring import score_search_summary, search_summary_beats_target


__all__ = [
    "ConstantSearchReport",
    "constant_search_to_dict",
    "render_constant_search_markdown",
    "run_constant_search",
    "score_search_summary",
    "search_summary_beats_target",
]


def run_constant_search(**kwargs) -> ConstantSearchReport:
    return run_auto_search(**kwargs)
