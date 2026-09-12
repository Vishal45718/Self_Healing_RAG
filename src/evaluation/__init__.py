"""Evaluation and Baseline Comparison package (Phase 7)."""

from src.evaluation.dataset import (
    DEFAULT_EVALUATION_DOCUMENTS,
    DEFAULT_EVALUATION_SAMPLES,
    load_default_dataset,
    load_default_documents,
    load_dataset_from_json,
)
from src.evaluation.metrics import (
    calculate_metric_deltas,
    calculate_system_metrics,
)
from src.evaluation.report import (
    generate_markdown_report,
    load_report_json,
    save_report_json,
)
from src.evaluation.runner import EvaluationRunner
from src.evaluation.schema import (
    EvalExecutionResult,
    EvalSample,
    EvaluationReport,
    ExpectedBehavior,
    ScenarioType,
    SystemMetrics,
)

__all__ = [
    "DEFAULT_EVALUATION_DOCUMENTS",
    "DEFAULT_EVALUATION_SAMPLES",
    "load_default_dataset",
    "load_default_documents",
    "load_dataset_from_json",
    "calculate_metric_deltas",
    "calculate_system_metrics",
    "generate_markdown_report",
    "load_report_json",
    "save_report_json",
    "EvaluationRunner",
    "EvalExecutionResult",
    "EvalSample",
    "EvaluationReport",
    "ExpectedBehavior",
    "ScenarioType",
    "SystemMetrics",
]
