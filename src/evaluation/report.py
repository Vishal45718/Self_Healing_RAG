"""Report generation and JSON serialization for Phase 7 evaluations."""

import json
from pathlib import Path
from src.evaluation.schema import EvaluationReport


def generate_markdown_report(report: EvaluationReport) -> str:
    """Generate a clean Markdown summary comparing Baseline vs Self-Healing RAG.

    Args:
        report: The completed EvaluationReport.

    Returns:
        Formatted Markdown report text.
    """
    bm = report.baseline_metrics
    sh = report.self_healing_metrics
    deltas = report.summary_deltas

    md = [
        "# RAG Evaluation & Baseline Comparison Report",
        "",
        f"- **Execution Timestamp**: `{report.timestamp}`",
        f"- **Dataset Size**: `{report.dataset_size}` test queries",
        "",
        "## 1. Executive Summary & Aggregate Metrics",
        "",
        "| Metric | Baseline RAG | Self-Healing RAG | Delta / Impact |",
        "|---|:---:|:---:|:---:|",
        f"| **Critic Pass Rate** | {bm.critic_pass_rate * 100:.1f}% | {sh.critic_pass_rate * 100:.1f}% | **{deltas.get('pass_rate_improvement', 0.0) * 100:+.1f}%** |",
        f"| **Groundedness Rate** | {bm.groundedness_rate * 100:.1f}% | {sh.groundedness_rate * 100:.1f}% | **{deltas.get('groundedness_improvement', 0.0) * 100:+.1f}%** |",
        f"| **Retrieval Sufficiency Rate** | {bm.retrieval_sufficiency_rate * 100:.1f}% | {sh.retrieval_sufficiency_rate * 100:.1f}% | **{deltas.get('sufficiency_improvement', 0.0) * 100:+.1f}%** |",
        f"| **Correct Abstention Rate** | {bm.correct_abstention_rate * 100:.1f}% | {sh.correct_abstention_rate * 100:.1f}% | **{deltas.get('abstention_improvement', 0.0) * 100:+.1f}%** |",
        f"| **Hallucination Rate (Unanswerable)** | {bm.hallucination_rate_on_unanswerable * 100:.1f}% | {sh.hallucination_rate_on_unanswerable * 100:.1f}% | {-(bm.hallucination_rate_on_unanswerable - sh.hallucination_rate_on_unanswerable) * 100:+.1f}% |",
        f"| **Recovery Success Rate** | N/A (0 retries) | {sh.recovery_success_rate * 100:.1f}% | {sh.recovery_success_rate * 100:.1f}% resolved |",
        f"| **Reformulation Success Rate** | N/A (0 reform) | {sh.reformulation_success_rate * 100:.1f}% | {sh.reformulation_success_rate * 100:.1f}% resolved |",
        f"| **Average Iterations / Retries** | 1.00 / 0.00 | {sh.average_iterations:.2f} / {sh.average_retries:.2f} | +{sh.average_retries:.2f} retries/query |",
        f"| **Average Latency (ms)** | {bm.average_latency_ms:.1f} ms | {sh.average_latency_ms:.1f} ms | {deltas.get('latency_overhead_ratio', 1.0):.2f}x overhead |",
        f"| **Average LLM Calls / Query** | {bm.average_llm_calls:.1f} | {sh.average_llm_calls:.1f} | {deltas.get('llm_call_overhead_ratio', 1.0):.2f}x overhead |",
        f"| **Total LLM Invocations** | {bm.total_llm_calls} | {sh.total_llm_calls} | +{sh.total_llm_calls - bm.total_llm_calls} calls |",
        "",
        "## 2. Key Findings & Insights",
        "",
        f"1. **Quality & Groundedness**: Self-Healing RAG achieved a Critic Pass Rate of **{sh.critic_pass_rate * 100:.1f}%** vs **{bm.critic_pass_rate * 100:.1f}%** for Baseline.",
        f"2. **Recovery Capability**: Of queries requiring self-healing recovery, **{sh.recovery_success_rate * 100:.1f}%** successfully reached verified passing/abstaining status.",
        f"3. **Cost-Quality Trade-off**: Self-Healing introduced an average of **{sh.average_retries:.2f} retries per query**, with a **{deltas.get('latency_overhead_ratio', 1.0):.2f}x** latency factor and **{deltas.get('llm_call_overhead_ratio', 1.0):.2f}x** LLM call overhead.",
        "",
    ]

    return "\n".join(md)


def save_report_json(report: EvaluationReport, output_path: str) -> None:
    """Serialize the evaluation report to a JSON file.

    Args:
        report: The EvaluationReport instance to save.
        output_path: Path where JSON will be written.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(report.model_dump_json(indent=2))


def load_report_json(input_path: str) -> EvaluationReport:
    """Load an EvaluationReport from a JSON file.

    Args:
        input_path: Filepath of the JSON report.

    Returns:
        Validated EvaluationReport.
    """
    path = Path(input_path)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return EvaluationReport.model_validate(data)
