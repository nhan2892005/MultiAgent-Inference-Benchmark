"""
heterogeneous_agents.py
=======================
Full heterogeneous multi-agent pipeline where each agent has its own:
  • Role  (orchestrator / code / guidance / video / synthesizer)
  • Model (Gemma 4 MoE, Qwen3-Coder, Llama Vision, …)
  • Engine (vLLM, SGLang, Ollama, llama.cpp)

Pipeline flow
-------------
  Student Query
       │
  [SupervisorActor]  ← uses Gemma 4 26B-MoE for planning
       │  Decomposes query → TaskManifest
       │
       ├── [CodeAgent]      Qwen3-Coder-8B  on SGLang
       ├── [GuidanceAgent]  Gemma 4 E4B     on Ollama
       └── [VideoAgent]     Llama3.2-Vision on llama.cpp
                │
           (timeout / cancel per agent)
                │
       [SynthesizerAgent]  Gemma 4 26B-MoE  on vLLM
                │
        Final Student Response

Run:
    python src/heterogeneous_agents.py
    python src/heterogeneous_agents.py --complexity high --dry-run
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import ray

sys.path.insert(0, str(Path(__file__).parent))
from agent_profiles import (
    AgentProfile, AgentRole, TaskComplexity,
    PROFILE_CATALOG, select_profile,
)
from unified_client import UnifiedAgentClient


# =============================================================================
# Task Manifest — Supervisor's decomposition of the student query
# =============================================================================

@dataclass
class TaskManifest:
    """
    Describes what each specialist sub-agent should do.
    Created by the Supervisor; consumed by the spawn loop.
    """
    original_query: str
    complexity: TaskComplexity
    # Per-role payloads (empty string = skip that agent)
    code_context:     str = ""   # Student's current code from IDE
    lecture_segment:  str = ""   # Relevant lecture transcript chunk
    learning_objectives: str = ""  # Module objectives for guidance agent

    def active_roles(self) -> list[AgentRole]:
        roles = [AgentRole.GUIDANCE]                            # Always active
        if self.code_context:     roles.append(AgentRole.CODE)
        if self.lecture_segment:  roles.append(AgentRole.VIDEO)
        return roles


# =============================================================================
# Base Sub-Agent Ray Actor
# =============================================================================

@ray.remote
class SubAgentActor:
    """
    Universal sub-agent actor.  Configured at construction time with an
    AgentProfile — so the same class handles Code, Guidance, Video, etc.

    The model + engine are completely determined by the profile; no
    engine-specific branching lives here.
    """

    def __init__(self, agent_id: str, profile: AgentProfile, simulate_slow: bool = False):
        self.agent_id      = agent_id
        self.profile       = profile
        self.simulate_slow = simulate_slow
        print(
            f"  ↳ Spawned [{agent_id}] "
            f"role={profile.role.value} "
            f"model={profile.model_id.split('/')[-1]} "
            f"engine={profile.engine_key}"
            + (" 💤 SLOW" if simulate_slow else "")
        )

    async def execute(self, task_input: str) -> dict:
        """
        Run inference against the role-specific model + engine.
        Returns a structured partial result dict.
        """
        t0 = time.perf_counter()

        if self.simulate_slow:
            print(f"  [{self.agent_id}] Simulating slow execution (20s)…")
            await asyncio.sleep(20)

        # ── Build UnifiedAgentClient from this profile's EngineConfig ─────────
        cfg    = self.profile.engine_config
        client = UnifiedAgentClient(cfg)

        # ── Role-specific prompt wrapper ──────────────────────────────────────
        user_prompt = self._wrap_prompt(task_input)
        result_text = ""
        error: Optional[str] = None

        try:
            if not await client.is_alive():
                # Engine unreachable → graceful mock (useful in --dry-run / CI)
                print(f"  [{self.agent_id}] ⚠️  Engine {cfg.base_url} unreachable — using mock.")
                result_text = self._mock_response()
            else:
                result_text, metrics = await client.stream_chat(
                    user_prompt=user_prompt,
                    system_prompt=self.profile.system_prompt,
                    request_id=hash(self.agent_id) % 10000,
                )
                print(
                    f"  [{self.agent_id}] ✅ done "
                    f"TTFT={metrics.ttft_ms:.0f}ms "
                    f"total={metrics.total_ms:.0f}ms "
                    f"tok={metrics.output_tokens}"
                )
        except Exception as exc:
            error = str(exc)
            print(f"  [{self.agent_id}] ❌ error: {error[:120]}")
        finally:
            await client.close()

        elapsed_ms = (time.perf_counter() - t0) * 1000
        parsed     = _parse_json_safe(result_text) if not error else {}

        return {
            "agent_id":   self.agent_id,
            "role":       self.profile.role.value,
            "model":      self.profile.model_id.split("/")[-1],
            "engine":     self.profile.engine_key,
            "elapsed_ms": round(elapsed_ms, 1),
            "error":      error,
            "output":     parsed,
            "raw":        result_text[:500],  # Truncated for logging
        }

    # ── Prompt wrappers per role ──────────────────────────────────────────────

    def _wrap_prompt(self, task_input: str) -> str:
        role = self.profile.role
        if role == AgentRole.CODE:
            return (
                f"Student question: {task_input}\n\n"
                "Inspect the code above for bugs and suggest fixes. "
                "Return JSON with keys: bugs (list), fixes (list), concepts (list)."
            )
        if role == AgentRole.GUIDANCE:
            return (
                f"Student question: {task_input}\n\n"
                "Provide 2–3 Socratic hints that guide without spoiling. "
                "Return JSON with keys: hints (list[str]), difficulty (float 1-10)."
            )
        if role == AgentRole.VIDEO:
            return (
                f"Student question: {task_input}\n\n"
                "Connect to the lecture content. "
                "Return JSON: references (list[str]), summary (str), key_concepts (list[str])."
            )
        return task_input

    def _mock_response(self) -> str:
        """Fallback when engine is offline — returns plausible dummy JSON."""
        role = self.profile.role
        if role == AgentRole.CODE:
            return json.dumps({
                "bugs": ["mock: engine offline"],
                "fixes": ["Restart inference engine"],
                "concepts": [],
            })
        if role == AgentRole.GUIDANCE:
            return json.dumps({
                "hints": ["Consider what happens when you apply H twice."],
                "difficulty": 5.0,
            })
        if role == AgentRole.VIDEO:
            return json.dumps({
                "references": ["Lecture 3, slide 12"],
                "summary": "Mock response — engine offline.",
                "key_concepts": ["superposition", "measurement"],
            })
        return json.dumps({"mock": True})


# =============================================================================
# Synthesizer Agent — its own model/engine, runs AFTER sub-agents complete
# =============================================================================

@ray.remote
class SynthesizerActor:

    def __init__(self, profile: AgentProfile):
        self.profile = profile
        print(
            f"  ↳ Spawned [Synthesizer] "
            f"model={profile.model_id.split('/')[-1]} "
            f"engine={profile.engine_key}"
        )

    async def synthesize(
        self,
        original_query: str,
        partial_results: list[dict],
    ) -> str:
        """Merge all partial results into one coherent student response."""
        # ── Build context block from partials ─────────────────────────────────
        context_lines = [f"Original question: {original_query}\n"]
        for pr in partial_results:
            if pr.get("error"):
                continue
            role   = pr["role"].upper()
            output = pr.get("output") or {}
            context_lines.append(f"[{role} agent output]:\n{json.dumps(output, indent=2)}")

        synthesis_prompt = (
            "\n\n".join(context_lines)
            + "\n\nSynthesize the above specialist outputs into one clear, "
              "pedagogically sound response for the student. "
              "Use Socratic framing where the code/guidance outputs align."
        )

        cfg    = self.profile.engine_config
        client = UnifiedAgentClient(cfg)
        try:
            if not await client.is_alive():
                return (
                    "[Synthesizer offline — raw partial results available in pipeline output]"
                )
            text, metrics = await client.stream_chat(
                user_prompt=synthesis_prompt,
                system_prompt=self.profile.system_prompt,
                request_id=9999,
            )
            print(
                f"  [Synthesizer] ✅ done "
                f"total={metrics.total_ms:.0f}ms tok={metrics.output_tokens}"
            )
            return text
        except Exception as exc:
            return f"[Synthesis error: {exc}]"
        finally:
            await client.close()


# =============================================================================
# Supervisor — the main orchestration actor
# =============================================================================

@ray.remote
class HeterogeneousSupervisor:
    """
    Orchestrates a fully heterogeneous multi-agent pipeline:

    1. select_profile()  — picks best model per role × complexity
    2. spawn actors      — each with its own engine/model
    3. concurrent run    — with per-agent timeout + ray.cancel
    4. synthesize        — a DIFFERENT (stronger) model merges results
    """

    def __init__(
        self,
        timeout_seconds: float = 12.0,
        slow_agent_index: Optional[int] = None,   # Which agent to make slow (demo)
    ):
        self.timeout = timeout_seconds
        self.slow_index = slow_agent_index

    async def run(
        self,
        query: str,
        manifest: TaskManifest,
    ) -> dict:
        t0 = time.perf_counter()

        print("\n" + "─" * 70)
        print(f"[Supervisor] Query: {query[:80]}")
        print(f"[Supervisor] Complexity: {manifest.complexity.value}")
        print(f"[Supervisor] Active roles: {[r.value for r in manifest.active_roles()]}")
        print("─" * 70)

        # ── 1. Select profiles for each active role ────────────────────────────
        role_profiles: list[tuple[AgentRole, AgentProfile]] = [
            (role, select_profile(role, manifest.complexity))
            for role in manifest.active_roles()
        ]

        print("\n[Supervisor] Agent roster:")
        for i, (role, prof) in enumerate(role_profiles):
            print(
                f"  [{i}] {role.value:<12} → "
                f"{prof.model_id.split('/')[-1]:<30} on {prof.engine_key}"
            )

        # ── 2. Spawn Ray actors ────────────────────────────────────────────────
        print("\n[Supervisor] 🚀 Spawning sub-agents…")
        actors = [
            SubAgentActor.remote(
                agent_id=f"{role.value}-{i}",
                profile=prof,
                simulate_slow=(i == self.slow_index),
            )
            for i, (role, prof) in enumerate(role_profiles)
        ]

        # ── 3. Build task inputs per role ─────────────────────────────────────
        task_inputs: list[str] = []
        for role, _ in role_profiles:
            if role == AgentRole.CODE:
                task_inputs.append(
                    f"{query}\n\n[Student Code]:\n{manifest.code_context or '# No code provided'}"
                )
            elif role == AgentRole.VIDEO:
                task_inputs.append(
                    f"{query}\n\n[Lecture Transcript Excerpt]:\n{manifest.lecture_segment or '(none)'}"
                )
            else:
                task_inputs.append(query)

        # ── 4. Dispatch concurrently ───────────────────────────────────────────
        futures = {
            i: actor.execute.remote(inp)
            for i, (actor, inp) in enumerate(zip(actors, task_inputs))
        }

        partial_results: list[dict] = []
        audit: list[dict] = []
        deadline = time.perf_counter() + self.timeout

        print(f"\n[Supervisor] ⏳ Collecting with timeout={self.timeout}s…")
        for i, future in futures.items():
            role_name = role_profiles[i][0].value
            remaining = max(deadline - time.perf_counter(), 0.5)
            try:
                result = await asyncio.wait_for(
                    asyncio.shield(future),   # shield prevents cancel from propagating
                    timeout=remaining,
                )
                partial_results.append(result)
                audit.append({"agent": f"{role_name}-{i}", "status": "success",
                               "elapsed_ms": result.get("elapsed_ms", 0)})
                print(f"  ✅ {role_name}-{i} collected.")
            except asyncio.TimeoutError:
                ray.cancel(future, force=True)
                audit.append({"agent": f"{role_name}-{i}", "status": "timeout"})
                print(f"  ⏰ {role_name}-{i} CANCELLED (timeout).")
            except Exception as exc:
                audit.append({"agent": f"{role_name}-{i}", "status": "error",
                               "error": str(exc)})
                print(f"  ❌ {role_name}-{i} error: {exc}")

        # ── 5. Synthesizer (different model — runs sequentially after gather) ──
        synth_profile = select_profile(AgentRole.SYNTHESIZER, manifest.complexity)
        print(
            f"\n[Supervisor] 🔀 Synthesising with "
            f"{synth_profile.model_id.split('/')[-1]} on {synth_profile.engine_key}…"
        )
        synthesizer = SynthesizerActor.remote(profile=synth_profile)
        final_text  = await synthesizer.synthesize.remote(query, partial_results)

        total_ms = (time.perf_counter() - t0) * 1000
        return {
            "query":          query,
            "complexity":     manifest.complexity.value,
            "final_response": final_text,
            "partial_results": partial_results,
            "audit": audit,
            "stats": {
                "total_agents":   len(role_profiles),
                "succeeded":      sum(1 for a in audit if a["status"] == "success"),
                "cancelled":      sum(1 for a in audit if a["status"] == "timeout"),
                "errored":        sum(1 for a in audit if a["status"] == "error"),
                "total_ms":       round(total_ms, 1),
            },
        }


# =============================================================================
# Output helpers
# =============================================================================

def print_pipeline_summary(output: dict) -> None:
    s = output["stats"]
    print("\n" + "═" * 72)
    print("  HETEROGENEOUS PIPELINE — SUMMARY")
    print("═" * 72)
    print(f"  Query      : {output['query'][:70]}")
    print(f"  Complexity : {output['complexity']}")
    print(f"  Agents     : {s['total_agents']} spawned | "
          f"{s['succeeded']} ✅ | {s['cancelled']} ⏰ | {s['errored']} ❌")
    print(f"  Wall time  : {s['total_ms']:.0f}ms")

    print("\n  Agent Roster:")
    for pr in output["partial_results"]:
        icon = "✅" if not pr.get("error") else "❌"
        print(
            f"    {icon} [{pr['role']:<10}] "
            f"{pr['model']:<30} via {pr['engine']:<10} "
            f"({pr['elapsed_ms']:.0f}ms)"
        )
    for a in output["audit"]:
        if a["status"] == "timeout":
            print(f"    ⏰ [{a['agent']}] cancelled (timeout)")

    print("\n  Final Synthesised Response:")
    print("  " + "─" * 60)
    for line in output["final_response"].split("\n")[:15]:
        print(f"  {line}")
    if output["final_response"].count("\n") > 15:
        print("  … (truncated)")
    print("═" * 72)


def _parse_json_safe(text: str) -> dict:
    text = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"raw_text": text[:300]}


# =============================================================================
# CLI entry-point
# =============================================================================

async def main(complexity_str: str, dry_run: bool) -> None:
    complexity = TaskComplexity(complexity_str)

    # ── Sample student interaction ─────────────────────────────────────────────
    query = (
        "My Qiskit circuit applies H then CNOT but the output probabilities "
        "are wrong — I expected |00⟩ and |11⟩ at 50% each but got |00⟩ at 100%. "
        "Can you help me debug this?"
    )
    manifest = TaskManifest(
        original_query=query,
        complexity=complexity,
        code_context=(
            "from qiskit import QuantumCircuit\n"
            "qc = QuantumCircuit(2)\n"
            "qc.h(0)\n"
            "qc.cx(0, 1)\n"
            "qc.measure_all()\n"
            "# Simulator: statevector_simulator\n"
            "# Getting |00⟩ = 1.0 — why?"
        ),
        lecture_segment=(
            "Lecture 4 — Entanglement: A Bell state is produced by applying H "
            "to qubit 0 then CNOT with qubit 0 as control and qubit 1 as target. "
            "The result is (|00⟩ + |11⟩)/√2. Common mistake: running on "
            "statevector_simulator after measure collapses the state."
        ),
    )

    print("╔══════════════════════════════════════════════════════════╗")
    print("║   Heterogeneous Multi-Agent Pipeline                    ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print(f"  Complexity : {complexity.value}")
    print(f"  Dry-run    : {dry_run} (engines won't be called if offline)")
    print()

    # ── Print selected profile roster without running ──────────────────────────
    print("📋  Profile Selection Table:")
    print(f"  {'Role':<14} {'Model':<35} {'Engine':<12} {'Max Tokens':>10}")
    print("  " + "─" * 74)
    for role in manifest.active_roles():
        prof = select_profile(role, complexity)
        print(
            f"  {role.value:<14} "
            f"{prof.model_id.split('/')[-1]:<35} "
            f"{prof.engine_key:<12} "
            f"{prof.max_tokens:>10}"
        )
    synth = select_profile(AgentRole.SYNTHESIZER, complexity)
    print(
        f"  {'synthesizer':<14} "
        f"{synth.model_id.split('/')[-1]:<35} "
        f"{synth.engine_key:<12} "
        f"{synth.max_tokens:>10}"
    )
    print()

    if dry_run:
        print("  [dry-run mode] Profile selection verified. Exiting without Ray.")
        return

    # ── Run Ray pipeline ───────────────────────────────────────────────────────
    print("🌐  Initialising Ray local cluster…")
    ray.init(num_cpus=6, ignore_reinit_error=True, log_to_driver=False)
    print(f"    Dashboard: {ray.get_webui_url() or 'http://localhost:8265'}\n")

    try:
        supervisor = HeterogeneousSupervisor.remote(
            timeout_seconds=12.0,
            slow_agent_index=None,   # Set to 0/1/2 to demo timeout
        )
        output = await supervisor.run.remote(query, manifest)
        print_pipeline_summary(output)

        out_path = Path("heterogeneous_output.json")
        out_path.write_text(json.dumps(output, indent=2, default=str))
        print(f"\n💾  Full output → {out_path}")

    finally:
        ray.shutdown()
        print("🔌  Ray shut down.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "--complexity", choices=["low", "medium", "high"], default="medium",
        help="Task complexity — controls which model each role gets"
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Only print the profile selection table; don't start Ray or call engines"
    )
    args = p.parse_args()
    asyncio.run(main(args.complexity, args.dry_run))