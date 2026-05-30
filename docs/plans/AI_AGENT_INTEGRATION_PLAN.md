# AI Agent Integration Plan

## Vision

Build an AI-powered research assistant that can autonomously run the
manifold partition pipeline, analyze intermediate results, make decisions
about parameter adjustments and continuation, and produce written reports
— all without manual intervention between steps.

The codebase restructure into `src/pipeline/`, `src/optimization/`,
`src/partition/`, `src/migration/`, and `src/mesh/` is complete.
`src/pipeline/pipeline_orchestrator.py` (`PipelineOrchestrator`) and
`src/pipeline/relaxation.py` (`run_relaxation()`) provide the
programmatic API that the AI agent layer calls.

---

## Architecture overview

```
┌─────────────────────────────────────────────────────────────┐
│  Layer 3: Batch Orchestration (FUTURE — only if needed)     │
│  ┌────────────────────────────────────────────┐             │
│  │  Apache Airflow / Prefect / cron           │             │
│  │  Schedules overnight sweeps, manages       │             │
│  │  retries, monitors long-running jobs       │             │
│  └─────────────────────┬──────────────────────┘             │
│                        │ triggers                           │
├────────────────────────┼────────────────────────────────────┤
│  Layer 2: AI Agent     │                                    │
│  ┌─────────────────────▼──────────────────────┐             │
│  │  LangGraph stateful workflow               │             │
│  │   ├─ Compute nodes (call src/ via tools)   │             │
│  │   ├─ Analyze nodes (LLM reads metrics)     │             │
│  │   ├─ Decide nodes  (LLM picks next step)   │             │
│  │   └─ Report nodes  (LLM writes summaries)  │             │
│  │                                             │             │
│  │  Uses LangChain tools to call Layer 1 ↓    │             │
│  └─────────────────────┬──────────────────────┘             │
│                        │ function calls                     │
├────────────────────────┼────────────────────────────────────┤
│  Layer 1: Compute Library (after restructure)               │
│  ┌─────────────────────▼──────────────────────┐             │
│  │  src/pipeline/pipeline_orchestrator.py      │             │
│  │  src/pipeline/io.py                         │             │
│  │  src/optimization/   src/partition/         │             │
│  │  src/migration/      src/mesh/              │             │
│  └────────────────────────────────────────────┘             │
└─────────────────────────────────────────────────────────────┘
```

**Layer 1** already exists (after restructure).  This plan covers
Layer 2 implementation and briefly addresses when Layer 3 becomes
relevant.

---

## Technology choices

### LangGraph (primary framework)

LangGraph models the pipeline as a directed graph where:
- **Nodes** are either Python functions (compute) or LLM calls (analyze/decide).
- **Edges** carry state (accumulated metrics, file paths, decision history).
- **Conditional edges** let the LLM choose which node to execute next.
- **Checkpointing** persists the full graph state, allowing resume after
  crashes or interruptions.

LangGraph is the right fit because the manifold partition pipeline is
inherently a multi-step workflow with conditional branching (retry with
different seed? switch solver? stop and report?).

### LangChain (underlying toolkit)

LangChain provides the components that LangGraph nodes use:
- **Tools**: typed Python functions the LLM can call.
- **Prompts**: structured templates for analysis and decision prompts.
- **Output parsers**: extract structured decisions (JSON) from LLM text.
- **LLM providers**: OpenAI, Anthropic, local models via Ollama, etc.

### LLM provider (configurable)

The agent should work with any LangChain-compatible LLM.  Recommended:
- **Development / learning**: GPT-4o or Claude via API (fast iteration).
- **Production / cost-sensitive**: local model via Ollama (e.g. Llama 3,
  Mistral) for routine analysis; cloud API for complex decisions.
- **Offline / cluster**: local model only.

### Apache Airflow (deferred)

Not needed until:
- Running on a multi-machine cluster.
- Managing 50+ concurrent parameter sweep jobs.
- Requiring a monitoring dashboard for overnight batch runs.

Lighter alternatives to consider first: **Prefect** (simpler setup than
Airflow, Python-native) or plain **cron + shell scripts** for scheduled
sweeps.

---

## Prerequisites

Before starting Layer 2 implementation:

1. **Codebase restructure complete.**  Done — `src/pipeline/` exposes
   the programmatic API: `PipelineOrchestrator.run_refinement_loop()`
   for Phase 2 refinement, and `run_relaxation()` in
   `src/pipeline/relaxation.py` for Phase 1 relaxation.

2. **Pipeline functions return structured results**, not just log output.
   Each function should return a dataclass or dict with all metrics the
   agent needs to make decisions:

   ```python
   @dataclass
   class RelaxationResult:
       energy: float
       initial_perimeter: float | None
       solution_path: str
       elapsed_seconds: float
       converged: bool
       n_partitions: int
       seed: int
       lambda_penalty: float

   @dataclass
   class RefinementResult:
       initial_perimeter: float
       final_perimeter: float
       perimeter_reduction_pct: float
       n_iterations: int
       converged: bool
       max_area_violation: float
       elapsed_seconds: float
       n_migrations: int
       output_path: str
       method: str
       exact_hessian: bool
   ```

3. **Python environment** with LangChain and LangGraph installed:
   ```bash
   pip install langchain langgraph langchain-openai  # or langchain-anthropic
   ```

---

## Implementation phases

### Phase 1 — LangChain tools (wrap the pipeline)

Create a set of LangChain tools that expose pipeline functions to the LLM.
Each tool is a thin wrapper that calls `src/pipeline/` and returns a
structured summary.

#### Tool inventory

| Tool name | Wraps | Input | Output |
|---|---|---|---|
| `run_relaxation` | `PipelineOrchestrator.run_relaxation()` | seed, lambda, n_partitions, mesh params | `RelaxationResult` as JSON |
| `run_refinement` | `PipelineOrchestrator.run_refinement()` | solution_path, method, max_iter, exact_hessian, lbfgs_memory | `RefinementResult` as JSON |
| `read_log` | file read | log_path, tail_lines | last N lines of log as text |
| `read_metadata` | YAML parse | run_directory | metadata dict as JSON |
| `list_results` | directory listing | — | list of result directories with timestamps |
| `compare_runs` | metric extraction | list of run directories | table of metrics side by side |
| `write_report` | file write | content, output_path | confirmation |

#### Example tool implementation

The exact constructor signatures and return types below must be
confirmed against the live code in `src/pipeline/` before implementing —
they are illustrative.  Phase 1 (relaxation) is the free function
`run_relaxation()` in `src/pipeline/relaxation.py`; Phase 2 (refinement)
is `PipelineOrchestrator.run_refinement_loop()` in
`src/pipeline/pipeline_orchestrator.py`.  Both `RelaxationConfig` and
`RefinementConfig` expose `from_yaml_dict()`, which accepts sectioned or
flat YAML; `run_refinement_loop()` returns a `dict`.

```python
from langchain.tools import tool

from src.pipeline.relaxation import run_relaxation as _run_relaxation
from src.pipeline.relaxation import RelaxationConfig
from src.pipeline.pipeline_orchestrator import (
    PipelineOrchestrator, RefinementConfig,
)

@tool
def run_relaxation(
    seed: int,
    lambda_penalty: float,
    n_partitions: int = 5,
    surface: str = "torus",
) -> dict:
    """Run Phase 1 Gamma-convergence relaxation.

    Returns energy, initial perimeter, solution path, and timing.
    Use this when you need to generate a new base solution from scratch.
    """
    config = RelaxationConfig.from_yaml_dict({
        "seed": seed,
        "lambda_penalty": lambda_penalty,
        "n_partitions": n_partitions,
        "surface": surface,
    })
    result = _run_relaxation(config)
    return _summarize_relaxation(result)   # → RelaxationResult-shaped dict


@tool
def run_refinement(
    solution_path: str,
    method: str = "ipopt",
    max_iterations: int = 1,
    exact_hessian: bool = False,
    lbfgs_memory: int = 20,
) -> dict:
    """Run perimeter refinement (contour extraction + IPOPT + migrations).

    Returns initial/final perimeter, iteration count, convergence status,
    and timing.  Use this after a relaxation run to optimize the partition
    boundary geometry.
    """
    config = RefinementConfig.from_yaml_dict({
        "method": method,
        "max_iterations": max_iterations,
        "exact_hessian": exact_hessian,
        "lbfgs_memory": lbfgs_memory,
    })
    orchestrator = PipelineOrchestrator(config)
    result = orchestrator.run_refinement_loop(solution_path)   # returns a dict
    return _summarize_refinement(result)   # → RefinementResult-shaped dict
```

#### Deliverables

- `agents/tools.py` — all tool definitions
- `agents/schemas.py` — result dataclasses (`RelaxationResult`,
  `RefinementResult`, etc.)
- Unit tests verifying tools return correct structure

---

### Phase 2 — LangGraph workflow (the decision engine)

Define the agent workflow as a LangGraph `StateGraph`.

#### Graph state

```python
from typing import TypedDict, Annotated
from langgraph.graph import add_messages

class PipelineState(TypedDict):
    # Accumulated chat history for the LLM
    messages: Annotated[list, add_messages]

    # Pipeline tracking
    current_phase: str            # "relaxation", "refinement", "report"
    run_directory: str | None
    solution_path: str | None

    # Metrics from completed steps
    relaxation_result: dict | None
    refinement_result: dict | None
    refinement_history: list[dict] # all refinement iterations

    # Agent decisions
    decision: str | None          # "continue", "retry", "adjust", "report"
    retry_count: int
    max_retries: int

    # Configuration
    config: dict                  # user-provided parameters
```

#### Graph topology

```
                    ┌──────────────┐
                    │  START       │
                    │  (load conf) │
                    └──────┬───────┘
                           │
                           ▼
                ┌────────────────────┐
           ┌───│  run_relaxation     │
           │   └─────────┬──────────┘
           │             │
           │             ▼
           │   ┌────────────────────┐
           │   │  analyze_relaxation│  ← LLM node
           │   │  (reads metrics)   │
           │   └─────────┬──────────┘
           │             │
           │             ▼
           │   ┌────────────────────┐
           │   │  decide_relaxation │  ← LLM node
           │   └────┬─────┬────────┘
           │        │     │
           │  ┌─────┘     └──────┐
           │  ▼                  ▼
           │ "retry"          "proceed"
           │  (new seed)         │
           │  │                  │
           └──┘                  ▼
                      ┌────────────────────┐
                 ┌───│  run_refinement     │
                 │   └─────────┬──────────┘
                 │             │
                 │             ▼
                 │   ┌────────────────────┐
                 │   │ analyze_refinement │  ← LLM node
                 │   └─────────┬──────────┘
                 │             │
                 │             ▼
                 │   ┌────────────────────┐
                 │   │ decide_refinement  │  ← LLM node
                 │   └──┬─────┬──────┬───┘
                 │      │     │      │
                 │ ┌────┘     │      └─────┐
                 │ ▼          ▼            ▼
                 │"continue" "adjust"   "report"
                 │ (more     (switch      │
                 │  iters)    solver)      │
                 └─┘          │            │
                              └──→ run_    │
                               refinement │
                                           ▼
                              ┌────────────────────┐
                              │  write_report      │  ← LLM node
                              └─────────┬──────────┘
                                        │
                                        ▼
                                     [END]
```

#### LLM node prompts

Each "analyze" and "decide" node has a prompt template.  Example for the
refinement analysis node:

```python
ANALYZE_REFINEMENT_PROMPT = """You are analyzing a perimeter refinement
optimization result for a {n_partitions}-cell partition on a torus.

## Metrics
- Initial perimeter: {initial_perimeter:.6f}
- Final perimeter: {final_perimeter:.6f}
- Reduction: {reduction_pct:.1f}%
- IPOPT iterations: {n_iterations}
- Converged: {converged}
- Max area violation: {max_area_violation:.2e}
- Wall time: {elapsed_seconds:.0f}s
- Method: {method}, exact_hessian={exact_hessian}
- Migrations triggered: {n_migrations}

## History of previous refinement iterations
{refinement_history}

## Your task
1. Assess whether the optimization converged to a satisfactory result.
2. Identify any warning signs (stalled objective, high area violation,
   excessive wall time).
3. Recommend one of:
   - "continue" — run another refinement iteration (perimeter still
     improving or migrations pending)
   - "adjust" — change solver settings (e.g. switch exact_hessian,
     change method, adjust tolerance) and re-run
   - "report" — optimization has converged or further improvement is
     unlikely; write a summary report

Respond with a JSON object:
{{"assessment": "...", "recommendation": "continue|adjust|report",
  "reason": "...", "adjustments": {{...}} }}
"""
```

#### LangGraph implementation skeleton

```python
from langgraph.graph import StateGraph, END

def build_pipeline_graph():
    graph = StateGraph(PipelineState)

    # Compute nodes (call tools, no LLM)
    graph.add_node("run_relaxation", run_relaxation_node)
    graph.add_node("run_refinement", run_refinement_node)

    # LLM analysis/decision nodes
    graph.add_node("analyze_relaxation", analyze_relaxation_node)
    graph.add_node("decide_relaxation", decide_relaxation_node)
    graph.add_node("analyze_refinement", analyze_refinement_node)
    graph.add_node("decide_refinement", decide_refinement_node)
    graph.add_node("write_report", write_report_node)

    # Edges
    graph.set_entry_point("run_relaxation")
    graph.add_edge("run_relaxation", "analyze_relaxation")
    graph.add_edge("analyze_relaxation", "decide_relaxation")

    graph.add_conditional_edges(
        "decide_relaxation",
        route_relaxation_decision,
        {
            "retry": "run_relaxation",
            "proceed": "run_refinement",
        },
    )

    graph.add_edge("run_refinement", "analyze_refinement")
    graph.add_edge("analyze_refinement", "decide_refinement")

    graph.add_conditional_edges(
        "decide_refinement",
        route_refinement_decision,
        {
            "continue": "run_refinement",
            "adjust": "run_refinement",
            "report": "write_report",
        },
    )

    graph.add_edge("write_report", END)

    return graph.compile(checkpointer=MemorySaver())
```

#### Deliverables

- `agents/graph.py` — the LangGraph workflow definition
- `agents/nodes.py` — individual node implementations
- `agents/prompts.py` — all prompt templates
- `agents/config.py` — LLM provider configuration
- Integration test: run the full graph on a small (coarse mesh) problem

---

### Phase 3 — Parameter sweep agent

Extend the graph with a parameter exploration mode that runs multiple
relaxation seeds or lambda values and selects the best candidate for
refinement.

> **Reuse the existing sweep tool.**  `sweep/parameter_sweep.py` already
> implements grid/paired sweep generation, local-sequential and
> local-parallel execution, `--resume`, and result collection into
> `experiment_index.yaml`.  The Phase 3 agent should *drive* that tool
> (or import its orchestration functions) rather than reimplement sweep
> mechanics.  The agent's added value is the LLM `plan_sweep` /
> `rank_candidates` nodes, not the sweep execution itself.

#### Extended graph

```
[START] → [plan_sweep] → [run_relaxation × N] → [rank_candidates]
                                                        │
                                                        ▼
                                              [select_best] → [run_refinement]
                                                                     │
                                                                    ...
                                                               (same as Phase 2)
```

The "plan_sweep" node is an LLM call that, given the user's goals
(target N, surface, quality requirements), proposes a set of
(seed, lambda) combinations to try.  The "rank_candidates" node
compares initial perimeters and selects the most promising one.

#### Parallelism

Multiple relaxation runs are independent and can run in parallel.
LangGraph supports this via `map` nodes or by spawning concurrent
branches.  For CPU-bound IPOPT work, this means either:
- Sequential on one machine (simplest).
- Parallel via Python `multiprocessing` (moderate effort).
- Distributed via Celery/Ray workers (future, with Airflow).

#### Deliverables

- `agents/sweep.py` — parameter sweep graph extension
- `agents/ranking.py` — candidate comparison logic
- Example: "find the best 5-partition on a torus, trying 5 seeds"

---

### Phase 4 — Report generation

The "write_report" node produces a structured markdown report
summarizing the full run.  Example output:

```markdown
# Partition Optimization Report
## Configuration
- Surface: torus (R=3, r=1)
- Partitions: 10
- Seed: 2234657, lambda: 3.25

## Phase 1: Relaxation
- Energy: -142.38
- Initial perimeter: 70.12
- Time: 45 min

## Phase 2: Refinement
- Method: IPOPT (L-BFGS, memory=20)
- Iterations: 3 topology cycles
- Final perimeter: 57.83 (17.5% reduction)
- Converged: yes (tol=1e-7)
- Total time: 48 min

## Agent Decisions
1. Relaxation result accepted (energy within expected range)
2. First refinement: switched from exact Hessian to L-BFGS after
   observing 15s/iter (expected <5s for L-BFGS)
3. Second refinement: continued (42 migrations pending)
4. Third refinement: stopped (0 migrations, perimeter stable)

## Recommendations
- Result quality: good (perimeter within 2% of 5-partition scaled
  estimate)
- Suggested next: try seeds [100, 200, 300] with lambda=3.0 to
  explore local minima
```

#### Deliverables

- `agents/report.py` — report generation node with template
- Example reports stored in `reports/` directory

---

## File structure

After full implementation, the agent layer lives in a dedicated
`agents/` directory at the project root:

```
agents/
├── __init__.py
├── config.py            (LLM provider settings, API keys via env vars)
├── schemas.py           (RelaxationResult, RefinementResult dataclasses)
├── tools.py             (LangChain tool definitions wrapping src/pipeline/)
├── prompts.py           (all LLM prompt templates)
├── nodes.py             (individual graph node implementations)
├── graph.py             (main LangGraph workflow definition)
├── sweep.py             (parameter sweep extension)
├── ranking.py           (candidate comparison logic)
├── report.py            (report generation)
├── run_agent.py         (CLI entry point: python agents/run_agent.py --config ...)
└── tests/
    ├── test_tools.py
    ├── test_graph.py
    └── test_prompts.py
```

---

## Dependencies

Add to `requirements.txt` (or a separate `requirements-agents.txt`):

```
langchain>=0.2
langgraph>=0.1
langchain-openai>=0.1       # or langchain-anthropic
pydantic>=2.0               # for structured output parsing
```

The agent layer is an optional add-on.  The core `src/` library has no
dependency on LangChain or LangGraph.

---

## Implementation timeline

| Phase | Depends on | Effort | Description |
|---|---|---|---|
| **Phase 1** | Restructure complete | 1 week | LangChain tools wrapping pipeline |
| **Phase 2** | Phase 1 | 1–2 weeks | LangGraph workflow with analyze/decide nodes |
| **Phase 3** | Phase 2 | 1 week | Parameter sweep agent |
| **Phase 4** | Phase 2 | 3–5 days | Report generation |
| **Layer 3** | Phase 3 + need | 1–2 weeks | Airflow/Prefect integration (only if needed) |

Total: approximately 4–6 weeks after the restructure, with a usable
agent (Phases 1–2) available after 2–3 weeks.

---

## Learning path

For someone new to LangChain/LangGraph, a suggested order:

1. **LangChain basics** (2–3 days): work through the official
   [LangChain tutorial](https://python.langchain.com/docs/tutorials/).
   Focus on: chat models, tools, output parsers, and agents.

2. **LangGraph basics** (2–3 days): work through the
   [LangGraph quickstart](https://langchain-ai.github.io/langgraph/tutorials/introduction/).
   Focus on: StateGraph, nodes, conditional edges, checkpointing.

3. **Build Phase 1** (this plan): implement the tools. This is the
   simplest step and gives immediate hands-on experience.

4. **Build Phase 2**: implement the graph. This is where LangGraph
   concepts (state, routing, checkpoints) come together.

---

## Risks and mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| LLM makes poor optimization decisions | Wasted compute time | Add hard-coded guardrails (max retries, timeout limits) alongside LLM decisions |
| LLM API costs for many analysis calls | Unexpected bills | Use local models (Ollama) for routine analysis; cloud API only for complex decisions |
| Pipeline functions are slow (hours) | Agent waits idle | LangGraph checkpointing allows resume; consider async execution |
| LLM halluccinates metric values | Wrong decisions | Always pass real metrics to the LLM; never ask it to "remember" numbers |
| LangChain/LangGraph API changes | Breaking updates | Pin versions in requirements; abstract tool definitions behind a thin adapter |

---

## Relationship to other plans

| Document | Relationship |
|---|---|
| `docs/reference/SCALABILITY_ANALYSIS.md` | **Informs agent decisions.** The agent should know when to switch from exact Hessian to L-BFGS based on partition count, and when to recommend coarser meshes. |
| `docs/math/03-analytical-steiner-derivatives` | **Improves compute layer (implemented).** The now-analytical exact-Hessian path gives faster, validated IPOPT iterations, so the agent can explore more parameter combinations in the same wall time. |
| `sweep/parameter_sweep.py` | **Existing tool to reuse.** The Phase 3 parameter-sweep agent should drive this tool rather than reimplement grid/paired sweep generation, parallel execution, and result collection. |
