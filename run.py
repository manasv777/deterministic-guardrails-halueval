"""Deterministic guardrail vs. plain LLM on HaluEval QA.

    python run.py run              # full 120-item experiment (cached; reruns are free)
    python run.py run --dry-run    # 6 items, end-to-end smoke test
    python run.py run --n 200      # different sample size (even number of items)
    python run.py report           # rebuild out/report.md from results/ (no API calls)
    python run.py dashboard        # rebuild out/dashboard.html from results/ (no API calls)
    python run.py fonts            # download the dashboard fonts into assets/fonts/
"""

from __future__ import annotations

import asyncio

import typer

from guardrail import config as C

app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.command()
def run(
    n: int = typer.Option(C.DEFAULT_N_ITEMS, help="Number of items (2 per sampled question)."),
    dry_run: bool = typer.Option(False, "--dry-run", help=f"Use {C.DRY_RUN_N_ITEMS} items."),
    seed: int = typer.Option(C.SEED, help="Sampling and bootstrap seed."),
    budget: float = typer.Option(C.BUDGET_USD, help="Abort if projected uncached cost exceeds this (USD)."),
) -> None:
    """Sample items, call the three configs (with caching), score, and write all outputs."""
    from guardrail.data import build_eval_set
    from guardrail.llm import estimate_cost, run_calls
    from guardrail.pipeline import score_items, summarize, write_outputs

    n_items = C.DRY_RUN_N_ITEMS if dry_run else n
    items = build_eval_set(n_items, seed)
    cfgs = C.configs()
    print(f"[run] {len(items)} items from {len(items) // 2} questions; seed {seed}")
    print(f"[run] models: small={C.SMALL_MODEL} large={C.LARGE_MODEL}; effort={C.EFFORT}")

    projected, n_calls = estimate_cost(cfgs, items)
    print(f"[budget] projected cost of {n_calls} uncached calls: ${projected:.4f} (limit ${budget:.2f})")
    if projected > budget:
        typer.echo(f"[budget] ABORT: projected ${projected:.2f} exceeds ${budget:.2f}", err=True)
        raise typer.Exit(1)

    records = asyncio.run(run_calls(cfgs, items))
    rows = score_items(items, records)
    summary = summarize(rows, seed)
    summary["run"]["dry_run"] = dry_run
    write_outputs(rows, summary)
    print(f"[run] actual cost of these {len(items)} items across configs: ${summary['run']['total_cost_usd']:.4f}")
    report()
    dashboard()


@app.command()
def report() -> None:
    """Rebuild out/report.md from results/ without calling the API."""
    from guardrail.pipeline import load_outputs
    from guardrail.report import write_report

    rows, summary = load_outputs()
    path = write_report(rows, summary)
    print(f"[report] wrote {path}")


@app.command()
def dashboard() -> None:
    """Rebuild out/dashboard.html from results/ without calling the API."""
    from guardrail.dashboard import write_dashboard
    from guardrail.pipeline import load_outputs

    rows, summary = load_outputs()
    path = write_dashboard(rows, summary)
    print(f"[dashboard] wrote {path}")


@app.command()
def fonts() -> None:
    """Download Switzer (Fontshare) and Geist Mono (Google Fonts) for embedding in the dashboard."""
    from guardrail.dashboard import fetch_fonts

    for p in fetch_fonts():
        print(f"[fonts] {p}")


if __name__ == "__main__":
    app()
