"""
ray_skeleton.py
===============
Ray-based Multi-Agent Skeleton: Spawn → Map → Timeout/Cancel → Reduce

Architecture
------------
  SupervisorActor  (Main Agent)
      │
      ├── SubAgentTask-0  (processes chunk_0 with shared context)
      ├── SubAgentTask-1  (processes chunk_1 with shared context)
      └── SubAgentTask-2  (processes chunk_2 with shared context)
              │
          (results gathered with timeout; slow agents cancelled)
              │
          ReduceActor  (merges partial JSON results into final output)

Run (after: pip install ray openai):
    python src/ray_skeleton.py
"""

from __future__ import annotations

import asyncio
import json
import random
import time
from pathlib import Path
from typing import Any, Optional

import ray

# ── Project imports (make UnifiedAgentClient available inside Ray workers) ──
import sys
sys.path.insert(0, str(Path(__file__).parent))
from unified_client import ENGINES, UnifiedAgentClient


# =============================================================================
# Shared Context — passed (by reference via object store) to every sub-agent
# =============================================================================

SHARED_CONTEXT = {
    "course":   "Graduate Quantum Information Science",
    "semester": "Spring 2026",
    "agent_system_instructions": (
        "You are a sub-agent in a distributed tutoring pipeline. "
        "Analyse the assigned student questions and return a JSON object "
        "with keys: questions_analysed (int), topics (list[str]), "
        "difficulty_avg (float 1-10), recommended_hints (list[str])."
    ),
    "model_config": {
        "engine":      "vllm",
        "max_tokens":  256,
        "temperature": 0.0,
    },
}

# ── Sample data to partition across sub-agents ────────────────────────────────
ALL_STUDENT_QUESTIONS: list[dict] = [
    {"id": f"q{i:03d}", "text": q}
    for i, q in enumerate([
        "Why does measuring a qubit collapse superposition?",
        "How does Grover's algorithm achieve quadratic speedup?",
        "What is the surface code threshold theorem?",
        "Explain the EPR paradox in simple terms.",
        "How do I implement a Toffoli gate in Qiskit?",
        "What is quantum teleportation — does information travel FTL?",
        "My quantum circuit has 500 gates. Is it too deep?",
        "Explain the difference between T-gate and S-gate.",
        "Why can't we clone an arbitrary quantum state?",
        "What does 'fidelity 0.99' mean for a gate operation?",
        "Derive the CHSH inequality from first principles.",
        "How does phase estimation work in Shor's algorithm?",
    ])
]


# =============================================================================
# Sub-Agent Actor (Ray remote class)
# =============================================================================

@ray.remote
class SubAgentTask:
    """
    Each instance processes an assigned partition of student questions.
    Communicates with a local inference engine via UnifiedAgentClient.

    Ray Remote class notes
    ----------------------
    • State is isolated per actor (each has its own Python process).
    • We re-create the AsyncOpenAI client inside the actor — it cannot be
      pickled across process boundaries.
    • Sleeping is used to simulate realistic LLM latency + inject one
      intentionally slow agent to demonstrate timeout/cancel.
    """

    def __init__(self, agent_id: int, shared_ctx: dict, slow: bool = False):
        self.agent_id   = agent_id
        self.shared_ctx = shared_ctx
        self.slow       = slow        # If True, this agent simulates timeout
        print(f"  [SubAgent-{agent_id}] Initialised {'(SLOW — will timeout)' if slow else ''}")

    async def run(self, questions: list[dict]) -> dict:
        """
        Map phase: analyse assigned questions using the inference engine.
        Returns a partial result dict.
        """
        t0 = time.perf_counter()
        print(f"  [SubAgent-{self.agent_id}] Starting — {len(questions)} questions assigned")

        # ── Simulate a slow agent for timeout demonstration ───────────────────
        if self.slow:
            print(f"  [SubAgent-{self.agent_id}] 💤 Simulating slow execution (15s delay)...")
            await asyncio.sleep(15)  # Will be cancelled by Supervisor

        # ── Build prompt ──────────────────────────────────────────────────────
        question_block = "\n".join(
            f"  {i+1}. [{q['id']}] {q['text']}"
            for i, q in enumerate(questions)
        )
        user_prompt = (
            f"Analyse these {len(questions)} student questions:\n{question_block}\n\n"
            "Return ONLY a valid JSON object with keys: "
            "questions_analysed (int), topics (list of strings), "
            "difficulty_avg (float), recommended_hints (list of strings)."
        )

        # ── Call inference engine ─────────────────────────────────────────────
        engine_key = self.shared_ctx["model_config"]["engine"]
        cfg        = ENGINES.get(engine_key, ENGINES["ollama"])
        client     = UnifiedAgentClient(cfg)

        result_text = ""
        try:
            result_text, metrics = await client.stream_chat(
                user_prompt=user_prompt,
                system_prompt=self.shared_ctx["agent_system_instructions"],
                request_id=self.agent_id,
            )
        finally:
            await client.close()

        # ── Parse JSON safely ─────────────────────────────────────────────────
        partial = _parse_json_safe(result_text)
        elapsed = (time.perf_counter() - t0) * 1000
        partial["_meta"] = {
            "agent_id":     self.agent_id,
            "questions_ids": [q["id"] for q in questions],
            "elapsed_ms":   round(elapsed, 1),
            "engine":       cfg.name,
        }

        print(f"  [SubAgent-{self.agent_id}] ✅ Done in {elapsed:.0f}ms")
        return partial


# =============================================================================
# Reduce Actor
# =============================================================================

@ray.remote
class ReduceActor:
    """
    Merges partial JSON outputs from successful sub-agents into a
    single consolidated result.
    """

    def reduce(self, partial_results: list[dict]) -> dict:
        print(f"\n  [ReduceActor] Merging {len(partial_results)} partial results...")

        all_topics:  list[str] = []
        all_hints:   list[str] = []
        total_q  = 0
        total_diff_sum   = 0.0
        valid_diff_count = 0
        agent_metas: list[dict] = []

        for pr in partial_results:
            meta = pr.get("_meta", {})
            agent_metas.append(meta)

            total_q += pr.get("questions_analysed", len(meta.get("questions_ids", [])))
            all_topics.extend(pr.get("topics", []))
            all_hints.extend(pr.get("recommended_hints", []))
            diff = pr.get("difficulty_avg")
            if diff is not None:
                total_diff_sum   += float(diff)
                valid_diff_count += 1

        reduced = {
            "total_questions_analysed": total_q,
            "unique_topics": sorted(set(all_topics)),
            "overall_difficulty_avg": round(
                total_diff_sum / max(valid_diff_count, 1), 2
            ),
            "all_hints": all_hints,
            "agents_contributed": [m.get("agent_id") for m in agent_metas],
            "total_elapsed_ms": sum(m.get("elapsed_ms", 0) for m in agent_metas),
        }

        print("  [ReduceActor] ✅ Reduction complete.")
        return reduced


# =============================================================================
# Supervisor Actor (Main Agent)
# =============================================================================

@ray.remote
class SupervisorActor:
    """
    Orchestrates the full Map → Cancel → Reduce pipeline.

    Pipeline steps
    --------------
    1. Partition ALL_STUDENT_QUESTIONS into N chunks (one per sub-agent).
    2. Spawn N SubAgentTask actors (one intentionally slow to test cancel).
    3. Wait for all with a timeout:
         - Agents that finish in time → collect result.
         - Agents that exceed timeout → ray.cancel() → skip.
    4. Pass successful results to ReduceActor for aggregation.
    5. Return final consolidated result + audit log.
    """

    def __init__(
        self,
        num_agents: int     = 3,
        timeout_seconds: float = 8.0,
    ):
        self.num_agents      = num_agents
        self.timeout_seconds = timeout_seconds
        print(f"[Supervisor] Initialised — {num_agents} sub-agents, timeout={timeout_seconds}s")

    async def orchestrate(self) -> dict:
        t_start = time.perf_counter()
        print(f"\n[Supervisor] 🚀 Starting orchestration pipeline")
        print(f"[Supervisor] Partitioning {len(ALL_STUDENT_QUESTIONS)} questions into "
              f"{self.num_agents} chunks...")

        # ── 1. Partition data (Map phase setup) ───────────────────────────────
        chunks = _partition(ALL_STUDENT_QUESTIONS, self.num_agents)
        for i, chunk in enumerate(chunks):
            ids = [q["id"] for q in chunk]
            print(f"  Chunk-{i}: {ids}")

        # ── 2. Put shared context into Ray object store (zero-copy sharing) ───
        ctx_ref = ray.put(SHARED_CONTEXT)

        # ── 3. Spawn sub-agents (last one is slow for timeout demo) ───────────
        print(f"\n[Supervisor] Spawning {self.num_agents} SubAgentTask actors...")
        agents = [
            SubAgentTask.remote(
                agent_id=i,
                shared_ctx=ray.get(ctx_ref),
                slow=(i == self.num_agents - 1),  # Last agent is intentionally slow
            )
            for i in range(self.num_agents)
        ]

        # ── 4. Dispatch run() concurrently (Map phase) ────────────────────────
        print("[Supervisor] 📡 Dispatching all sub-agents concurrently (Map phase)...")
        futures = {
            i: agent.run.remote(chunks[i])
            for i, agent in enumerate(agents)
        }

        # ── 5. Collect with per-agent timeout (Cancel phase) ──────────────────
        partial_results: list[dict] = []
        audit_log:       list[dict] = []

        deadline = time.perf_counter() + self.timeout_seconds

        # We poll futures one-by-one with remaining time budget
        for agent_id, future in futures.items():
            remaining = max(deadline - time.perf_counter(), 0.1)
            print(f"\n[Supervisor] ⏳ Waiting for SubAgent-{agent_id} "
                  f"(up to {remaining:.1f}s remaining)...")
            try:
                result = await asyncio.wait_for(
                    _ray_future_to_coroutine(future),
                    timeout=remaining,
                )
                partial_results.append(result)
                audit_log.append({
                    "agent_id": agent_id,
                    "status":   "success",
                    "elapsed_ms": result.get("_meta", {}).get("elapsed_ms", 0),
                })
                print(f"[Supervisor] ✅ SubAgent-{agent_id} completed successfully.")

            except asyncio.TimeoutError:
                print(f"[Supervisor] ⏰ SubAgent-{agent_id} exceeded timeout — cancelling.")
                ray.cancel(future, force=True)
                audit_log.append({
                    "agent_id": agent_id,
                    "status":   "cancelled_timeout",
                    "timeout_seconds": self.timeout_seconds,
                })

            except Exception as exc:
                print(f"[Supervisor] ❌ SubAgent-{agent_id} failed: {exc}")
                audit_log.append({
                    "agent_id": agent_id,
                    "status":   "error",
                    "error":    str(exc),
                })

        # ── 6. Reduce phase ───────────────────────────────────────────────────
        print(f"\n[Supervisor] 🔀 Running Reduce phase "
              f"({len(partial_results)}/{self.num_agents} partial results)...")

        reducer = ReduceActor.remote()
        final   = ray.get(reducer.reduce.remote(partial_results))

        total_elapsed = (time.perf_counter() - t_start) * 1000
        output = {
            "final_result":   final,
            "audit_log":      audit_log,
            "pipeline_stats": {
                "num_agents":      self.num_agents,
                "timeout_seconds": self.timeout_seconds,
                "agents_success":  len(partial_results),
                "agents_cancelled": sum(1 for a in audit_log if a["status"] == "cancelled_timeout"),
                "agents_errored":  sum(1 for a in audit_log if a["status"] == "error"),
                "total_elapsed_ms": round(total_elapsed, 1),
            },
        }
        return output


# =============================================================================
# Helpers
# =============================================================================

def _partition(items: list, n: int) -> list[list]:
    """Split a list into n roughly-equal chunks."""
    k, rem = divmod(len(items), n)
    result, start = [], 0
    for i in range(n):
        end = start + k + (1 if i < rem else 0)
        result.append(items[start:end])
        start = end
    return result


def _parse_json_safe(text: str) -> dict:
    """Extract first JSON object from a string, returning {} on failure."""
    # Strip markdown fences if present
    text = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Fallback: return a mock result so Reduce doesn't break
        return {
            "questions_analysed": 0,
            "topics":             ["parse_error"],
            "difficulty_avg":     5.0,
            "recommended_hints":  ["Could not parse agent output — check raw response."],
            "_raw": text[:200],
        }


async def _ray_future_to_coroutine(future: Any) -> Any:
    """
    Bridge between Ray's ObjectRef and asyncio.
    Ray 2.x supports `await` on ObjectRef directly in async context,
    but we wrap in asyncio.wait_for for clean timeout handling.
    """
    return await future


# =============================================================================
# Pretty-print output
# =============================================================================

def print_pipeline_output(output: dict) -> None:
    print("\n" + "═" * 72)
    print("  PIPELINE RESULTS")
    print("═" * 72)

    stats = output["pipeline_stats"]
    print(f"\n  🏁 Pipeline Stats")
    print(f"     Agents:    {stats['num_agents']} total | "
          f"{stats['agents_success']} succeeded | "
          f"{stats['agents_cancelled']} cancelled | "
          f"{stats['agents_errored']} errored")
    print(f"     Wall-time: {stats['total_elapsed_ms']:.0f}ms")
    print(f"     Timeout:   {stats['timeout_seconds']}s per agent")

    print(f"\n  📋 Audit Log")
    for entry in output["audit_log"]:
        icon = {"success": "✅", "cancelled_timeout": "⏰", "error": "❌"}.get(
            entry["status"], "❓"
        )
        print(f"     {icon} Agent-{entry['agent_id']}: {entry['status']}")

    final = output["final_result"]
    print(f"\n  📊 Reduced Result")
    print(f"     Questions analysed:  {final.get('total_questions_analysed', 'N/A')}")
    print(f"     Unique topics:       {final.get('unique_topics', [])}")
    print(f"     Avg difficulty:      {final.get('overall_difficulty_avg', 'N/A')}/10")
    print(f"     Agents contributed:  {final.get('agents_contributed', [])}")
    hints = final.get("all_hints", [])
    if hints:
        print(f"\n  💡 Collected Hints ({len(hints)})")
        for h in hints[:5]:
            print(f"     • {h[:90]}")
        if len(hints) > 5:
            print(f"     ... and {len(hints)-5} more")

    print("\n" + "═" * 72)

    # ── Save to file ──────────────────────────────────────────────────────────
    out_path = Path("ray_pipeline_output.json")
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\n💾  Full output saved → {out_path}")


# =============================================================================
# Entry-point
# =============================================================================

async def main() -> None:
    # ── Initialise Ray (local cluster) ────────────────────────────────────────
    print("🌐  Initialising Ray local cluster...")
    ray.init(
        num_cpus=4,
        ignore_reinit_error=True,
        log_to_driver=False,   # Suppress worker logs; we print our own
    )
    print(f"    Dashboard: {ray.get_webui_url() or 'http://localhost:8265'}")

    try:
        # ── Create and run Supervisor ─────────────────────────────────────────
        supervisor = SupervisorActor.remote(
            num_agents=3,
            timeout_seconds=8.0,   # Agent-2 is slow (15s) → will be cancelled
        )
        output = await supervisor.orchestrate.remote()
        print_pipeline_output(output)

    finally:
        ray.shutdown()
        print("\n🔌  Ray cluster shut down.")


if __name__ == "__main__":
    asyncio.run(main())