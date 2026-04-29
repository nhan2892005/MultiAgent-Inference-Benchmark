"""
benchmark.py
============
Multi-Agent Context Benchmark Suite
-------------------------------------
Measures TTFT, TPOT, and total latency across 4 inference engines,
simulating the shared-context pattern that dominates multi-agent workloads.

Key scenario replicated
-----------------------
  - ONE large system prompt (tool definitions + shared context, ~2500 tokens)
  - N concurrent user requests with short, distinct questions
  - Prefix-caching engines (vLLM PagedAttention, SGLang RadixAttention)
    should show dramatically lower TTFT from request #2 onward.

Run:
    python src/benchmark.py [--engines ollama vllm] [--concurrency 15] [--runs 3]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# ── Add project root to path ──────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from unified_client import ENGINES, EngineConfig, RequestMetrics, UnifiedAgentClient


# =============================================================================
# 1. Shared Context Generator (~2500 tokens)
# =============================================================================

def build_shared_system_prompt() -> str:
    """
    Builds a realistic, large system prompt that simulates what a multi-agent
    orchestrator sends to every sub-agent.  Covers:
      • Agent identity & role
      • Available tools (JSON schemas)
      • Domain context (academic STEM tutoring data)
      • Response format rules
    This is intentionally verbose to stress prefix-caching mechanisms.
    """
    tool_schema_block = """
## Available Tools

### Tool: search_knowledge_base
```json
{
  "name": "search_knowledge_base",
  "description": "Semantic search over the course lecture corpus. Returns top-k chunks.",
  "parameters": {
    "type": "object",
    "properties": {
      "query": {"type": "string", "description": "Natural-language search query"},
      "top_k": {"type": "integer", "default": 5, "description": "Number of chunks to return"},
      "module_filter": {"type": "string", "description": "Optional: restrict to a specific module (e.g. 'quantum_gates')"}
    },
    "required": ["query"]
  }
}
```

### Tool: execute_code
```json
{
  "name": "execute_code",
  "description": "Run Python code in an isolated sandbox. Returns stdout, stderr, and exit code.",
  "parameters": {
    "type": "object",
    "properties": {
      "code": {"type": "string", "description": "Python source code to execute"},
      "timeout_seconds": {"type": "integer", "default": 10},
      "packages": {"type": "array", "items": {"type": "string"}, "description": "pip packages to install before running"}
    },
    "required": ["code"]
  }
}
```

### Tool: retrieve_student_context
```json
{
  "name": "retrieve_student_context",
  "description": "Returns the student's recent interaction history, current code, and learning objectives.",
  "parameters": {
    "type": "object",
    "properties": {
      "student_id": {"type": "string"},
      "include_code": {"type": "boolean", "default": true},
      "include_history_turns": {"type": "integer", "default": 5}
    },
    "required": ["student_id"]
  }
}
```

### Tool: submit_hint
```json
{
  "name": "submit_hint",
  "description": "Deliver a Socratic hint to the student without revealing the full solution.",
  "parameters": {
    "type": "object",
    "properties": {
      "hint_text": {"type": "string"},
      "difficulty": {"type": "string", "enum": ["gentle", "medium", "direct"]},
      "reference_concept": {"type": "string", "description": "Curriculum concept this hint addresses"}
    },
    "required": ["hint_text", "difficulty"]
  }
}
```
"""

    domain_context_block = """
## Course Context: Graduate Quantum Information Science

This course covers quantum computing and quantum information theory at the graduate level.
Key modules:
  1. Linear Algebra for Quantum Mechanics (Hilbert spaces, tensor products, bra-ket notation)
  2. Quantum Gates & Circuits (Hadamard, CNOT, Toffoli, phase gates, universality theorems)
  3. Quantum Algorithms (Deutsch-Jozsa, Grover's search O(√N), Shor's factoring O(log³N))
  4. Quantum Error Correction (stabiliser codes, surface codes, fault-tolerance thresholds)
  5. Quantum Entanglement & Bell Inequalities (EPR pairs, CHSH game, teleportation protocol)

Common student misconceptions to watch for:
  - Confusing quantum superposition with classical probability distributions
  - Assuming measurement is reversible (it is NOT — wavefunction collapse is irreversible)
  - Believing all quantum gates must be reversible (measurement gates are not unitary)
  - Incorrectly applying the no-cloning theorem to mixed states
  - Confusing qubit fidelity with circuit depth optimisation

Pedagogical approach: Socratic method. Guide students to the answer through leading questions.
NEVER give a direct answer to a debugging problem — help the student discover it themselves.
Always relate abstract concepts back to a physical intuition (e.g., spin of a particle).
"""

    format_rules_block = """
## Response Format Rules

1. **Language**: Always respond in the same language as the student's question.
2. **Length**: Keep responses under 300 words unless a derivation is explicitly requested.
3. **Code**: Wrap all code in triple-backtick blocks with the language specified.
4. **Equations**: Use LaTeX inline ($...$) for mathematical expressions.
5. **Tool calls**: When invoking a tool, output ONLY the JSON tool-call object on a single line, prefixed with TOOL_CALL:.
6. **Confidence**: If you are less than 80% confident in a factual claim, preface it with "I believe..." and recommend the student verify via the course textbook.
7. **Citations**: When referencing lecture content, cite the module number and slide title if known.
8. **Tone**: Encouraging but academically rigorous. Avoid oversimplification.
9. **Handoff**: If a question is outside your specialist domain, end your response with HANDOFF: <domain>.
"""

    agent_identity_block = """
## Your Role: Specialist Sub-Agent

You are one of several specialist agents in a multi-agent intelligent tutoring system (ITAS).
Your responses feed into a Synthesizer Agent that merges outputs from parallel specialists.

Specialist domains available in this system:
  - VIDEO_AGENT: Answers questions by referencing the current lecture video transcript.
  - CODE_AGENT:  Inspects the student's live IDE code for bugs and conceptual errors.
  - GUIDANCE_AGENT: Provides Socratic hints aligned with module learning objectives.
  - SYNTHESIZER: Merges outputs from all three specialists (this is NOT your role).

You must complete your analysis fully before returning. Do NOT wait for other agents.
Your output will be consumed programmatically — adhere strictly to the format rules above.
"""

    # Assemble the full prompt (~2400-2700 tokens)
    return "\n".join([
        "# ITAS Multi-Agent System — Specialist Agent Instructions",
        agent_identity_block,
        tool_schema_block,
        domain_context_block,
        format_rules_block,
        "---",
        "Begin processing the student's question below.",
    ])


# =============================================================================
# 2. Diverse user questions (short — isolates system-prompt prefix overhead)
# =============================================================================

USER_QUESTIONS: list[str] = [
    "Why does measuring a qubit destroy the superposition?",
    "My Hadamard gate code gives wrong output. What should I check first?",
    "Can you explain the intuition behind Grover's quadratic speedup?",
    "What is a stabiliser code and how does it detect errors?",
    "Is it possible to copy an unknown quantum state? Why or why not?",
    "How do I implement CNOT in Qiskit?",
    "What is quantum entanglement intuitively, without the math?",
    "My circuit has depth 200 — is that too deep for current hardware?",
    "Explain the difference between a pure state and a mixed state.",
    "What is the threshold theorem for fault-tolerant quantum computation?",
    "How does quantum teleportation not violate the no-cloning theorem?",
    "My measurement probabilities don't sum to 1. What went wrong?",
    "Give me a hint on implementing Deutsch-Jozsa without spoiling the answer.",
    "What's the relationship between tensor products and multi-qubit systems?",
    "Explain phase kickback in the context of oracle queries.",
    "How does surface code improve upon the 3-qubit repetition code?",
    "My QuantumCircuit gives a DeprecationWarning on transpile. How do I fix it?",
    "What does 'fidelity of 0.95' mean for a two-qubit gate?",
    "Can classical information be transmitted faster than light using entanglement?",
    "Walk me through why Shor's algorithm needs quantum Fourier transform.",
]


# =============================================================================
# 3. Benchmark runner
# =============================================================================

@dataclass
class BenchmarkResult:
    engine: str
    concurrency: int
    total_requests: int
    successful: int
    failed: int
    # Latency distributions (ms)
    ttft_values: list[float] = field(default_factory=list)
    tpot_values: list[float] = field(default_factory=list)
    total_values: list[float] = field(default_factory=list)
    wall_clock_ms: float      = 0.0

    @property
    def throughput_rps(self) -> float:
        """Successful requests per second (wall-clock based)."""
        return (self.successful / self.wall_clock_ms * 1000) if self.wall_clock_ms else 0

    def percentile(self, values: list[float], p: float) -> float:
        if not values:
            return 0.0
        sorted_v = sorted(values)
        idx = int(len(sorted_v) * p / 100)
        return sorted_v[min(idx, len(sorted_v) - 1)]


async def run_single_engine_benchmark(
    config: EngineConfig,
    concurrency: int,
    run_index: int,
) -> BenchmarkResult:
    """
    Fire `concurrency` requests simultaneously to a single engine,
    all sharing the same system prompt (tests prefix-cache effectiveness).
    """
    print(f"  ▶  [{config.name}] run={run_index+1} | concurrency={concurrency}")

    system_prompt = build_shared_system_prompt()
    questions     = (USER_QUESTIONS * 10)[:concurrency]  # Repeat if needed
    client        = UnifiedAgentClient(config)

    # ── Check liveness ────────────────────────────────────────────────────────
    if not await client.is_alive():
        print(f"  ⚠️   [{config.name}] is not reachable — skipping.")
        await client.close()
        return BenchmarkResult(
            engine=config.name, concurrency=concurrency,
            total_requests=concurrency, successful=0, failed=concurrency,
        )

    # ── Concurrent dispatch ───────────────────────────────────────────────────
    async def single_request(req_id: int, question: str) -> RequestMetrics:
        _, metrics = await client.stream_chat(
            user_prompt=question,
            system_prompt=system_prompt,
            request_id=req_id,
        )
        return metrics

    t_wall_start = time.perf_counter()
    tasks        = [
        asyncio.create_task(single_request(i, q))
        for i, q in enumerate(questions)
    ]
    all_metrics: list[RequestMetrics] = await asyncio.gather(*tasks, return_exceptions=False)
    wall_ms = (time.perf_counter() - t_wall_start) * 1000

    await client.close()

    # ── Aggregate ─────────────────────────────────────────────────────────────
    ok  = [m for m in all_metrics if m.ok]
    err = [m for m in all_metrics if not m.ok]

    result = BenchmarkResult(
        engine=config.name,
        concurrency=concurrency,
        total_requests=concurrency,
        successful=len(ok),
        failed=len(err),
        ttft_values=[m.ttft_ms   for m in ok],
        tpot_values=[m.tpot_ms   for m in ok],
        total_values=[m.total_ms for m in ok],
        wall_clock_ms=wall_ms,
    )

    if err:
        print(f"  ⚠️   [{config.name}] {len(err)} errors: {err[0].error[:80]}")

    return result


async def run_benchmark(
    engine_keys: list[str],
    concurrency: int,
    num_runs: int,
) -> list[BenchmarkResult]:
    """Run benchmark for each engine; average over multiple runs."""
    all_results: list[BenchmarkResult] = []

    for key in engine_keys:
        if key not in ENGINES:
            print(f"⚠️   Unknown engine '{key}', skipping.")
            continue

        config  = ENGINES[key]
        run_results: list[BenchmarkResult] = []

        print(f"\n🔬  Benchmarking: {config.name}")
        for run in range(num_runs):
            r = await run_single_engine_benchmark(config, concurrency, run)
            run_results.append(r)
            if num_runs > 1:
                await asyncio.sleep(2)   # Let KV cache cool between runs

        # Merge runs
        merged = BenchmarkResult(
            engine=config.name,
            concurrency=concurrency,
            total_requests=sum(r.total_requests for r in run_results),
            successful=sum(r.successful for r in run_results),
            failed=sum(r.failed for r in run_results),
            ttft_values =[v for r in run_results for v in r.ttft_values],
            tpot_values =[v for r in run_results for v in r.tpot_values],
            total_values=[v for r in run_results for v in r.total_values],
            wall_clock_ms=statistics.mean(r.wall_clock_ms for r in run_results),
        )
        all_results.append(merged)

    return all_results


# =============================================================================
# 4. Results display
# =============================================================================

def print_results_table(results: list[BenchmarkResult]) -> None:
    """Pretty-print a comparison table to stdout."""
    SEP  = "─" * 112
    print(f"\n{'═'*112}")
    print("  MULTI-AGENT CONTEXT BENCHMARK — RESULTS")
    print(f"{'═'*112}")
    print(
        f"  {'Engine':<14} {'Concur':>6} {'OK':>4} {'Err':>4} "
        f"{'TTFT p50':>10} {'TTFT p95':>10} "
        f"{'TPOT p50':>10} "
        f"{'Total p50':>10} {'Total p95':>10} "
        f"{'RPS':>7}"
    )
    print(f"  {SEP}")

    for r in results:
        p50_ttft  = r.percentile(r.ttft_values,  50)
        p95_ttft  = r.percentile(r.ttft_values,  95)
        p50_tpot  = r.percentile(r.tpot_values,  50)
        p50_total = r.percentile(r.total_values, 50)
        p95_total = r.percentile(r.total_values, 95)
        rps       = r.throughput_rps

        print(
            f"  {r.engine:<14} {r.concurrency:>6} {r.successful:>4} {r.failed:>4} "
            f"{p50_ttft:>8.0f}ms {p95_ttft:>8.0f}ms "
            f"{p50_tpot:>8.1f}ms "
            f"{p50_total:>8.0f}ms {p95_total:>8.0f}ms "
            f"{rps:>6.1f}"
        )

    print(f"  {SEP}")
    print("  TTFT=Time-To-First-Token  TPOT=Time-Per-Output-Token  RPS=successful req/s")
    print(f"{'═'*112}\n")

    # ── Prefix-cache insight ──────────────────────────────────────────────────
    print("📊  PREFIX-CACHE INSIGHT")
    print("  Engines with KV-cache prefix reuse (vLLM PagedAttention / SGLang RadixAttention)")
    print("  should show LOWER TTFT p95 vs p50 gap (less tail latency variance from cache hits).\n")


def save_results_json(results: list[BenchmarkResult], path: str = "benchmark_results.json") -> None:
    data = [
        {
            "engine":        r.engine,
            "concurrency":   r.concurrency,
            "successful":    r.successful,
            "failed":        r.failed,
            "ttft_p50_ms":   r.percentile(r.ttft_values,  50),
            "ttft_p95_ms":   r.percentile(r.ttft_values,  95),
            "tpot_p50_ms":   r.percentile(r.tpot_values,  50),
            "total_p50_ms":  r.percentile(r.total_values, 50),
            "total_p95_ms":  r.percentile(r.total_values, 95),
            "throughput_rps": r.throughput_rps,
            "raw_ttft_ms":   r.ttft_values,
        }
        for r in results
    ]
    Path(path).write_text(json.dumps(data, indent=2))
    print(f"💾  Raw results saved → {path}")


# =============================================================================
# 5. CLI entry-point
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Multi-Agent Inference Engine Benchmark")
    p.add_argument(
        "--engines", nargs="+",
        default=list(ENGINES.keys()),
        choices=list(ENGINES.keys()),
        help="Which engines to benchmark (default: all)",
    )
    p.add_argument(
        "--concurrency", type=int, default=15,
        help="Number of simultaneous requests (default: 15)",
    )
    p.add_argument(
        "--runs", type=int, default=2,
        help="Benchmark runs per engine (results averaged, default: 2)",
    )
    p.add_argument(
        "--output", default="benchmark_results.json",
        help="Output JSON file path",
    )
    return p.parse_args()


async def main() -> None:
    args = parse_args()

    print("╔══════════════════════════════════════════════════════════╗")
    print("║    Multi-Agent Context Benchmark Suite                  ║")
    print("╚══════════════════════════════════════════════════════════╝")
    prompt_words = build_shared_system_prompt().split()
    print(f"  System prompt: ~{len(prompt_words)} words (~{len(prompt_words)//0.75:.0f} tokens estimated)")
    print(f"  Engines:       {args.engines}")
    print(f"  Concurrency:   {args.concurrency} simultaneous requests")
    print(f"  Runs:          {args.runs} per engine\n")

    t_total = time.perf_counter()
    results = await run_benchmark(args.engines, args.concurrency, args.runs)
    elapsed = time.perf_counter() - t_total

    print_results_table(results)
    save_results_json(results, args.output)
    print(f"⏱️   Total benchmark time: {elapsed:.1f}s")


if __name__ == "__main__":
    asyncio.run(main())