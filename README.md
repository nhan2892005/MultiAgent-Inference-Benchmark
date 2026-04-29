# Multi-Agent Inference Benchmark & Ray Skeleton

A production-ready benchmark harness and distributed multi-agent skeleton for evaluating local LLM inference engines under **shared-context, high-concurrency** workloads. Designed for researchers and engineers running heterogeneous agent pipelines with mixed model families and inference backends.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Quickstart](#quickstart)
- [Inference Engines](#inference-engines)
- [Benchmark Suite](#benchmark-suite)
- [Ray Multi-Agent Pipeline](#ray-multi-agent-pipeline)
- [Heterogeneous Agent Pipeline](#heterogeneous-agent-pipeline)
- [Agent Profiles & Model Registry](#agent-profiles--model-registry)
- [Configuration](#configuration)
- [Expected Results](#expected-results)
- [Key Concepts](#key-concepts)
- [Troubleshooting](#troubleshooting)

---

## Overview

This project addresses a core challenge in production multi-agent systems: **how do different inference engines perform when many agents share a large common context (e.g. a system prompt with tool definitions)?**

It provides two complementary components:

| Component | File | Purpose |
|---|---|---|
| Benchmark Suite | `benchmark.py` | Measures TTFT, TPOT, and throughput across 4 engines under concurrent load |
| Ray Skeleton | `ray_skeleton.py` | Demonstrates Supervisor → SubAgent → Reduce pattern with timeout/cancel |
| Heterogeneous Pipeline | `heterogeneous_agents.py` | Full pipeline with per-role model + engine selection |
| Agent Profiles | `agent_profiles.py` | Model registry, profile catalog, and routing table |
| Unified Client | `unified_client.py` | Single OpenAI-compatible interface over all 4 engines |

---

## Architecture

### Scope 1 & 2 — Benchmark Layer

```
UnifiedAgentClient  →  OpenAI-compatible API
                              │
          ┌───────────────────┼───────────────────┐
          ▼           ▼       ▼         ▼
       Ollama    llama.cpp   vLLM    SGLang
       :11434    :8080       :8000   :30000

benchmark.py
 ├── build_shared_system_prompt()  ~2500 tokens
 ├── 15 concurrent requests (one per question)
 ├── Measure: TTFT, TPOT, Total Latency
 └── Print results table + save JSON
```

### Scope 3 — Ray Multi-Agent Layer

```
SupervisorActor (Main Agent)
     │
     ├── ray.put(shared_context)  → Object Store
     │
     ├── SubAgentTask-0  [questions 0-3]   ✅ fast
     ├── SubAgentTask-1  [questions 4-7]   ✅ fast
     └── SubAgentTask-2  [questions 8-11]  ⏰ SLOW → CANCEL
     
     asyncio.wait_for(future, timeout=8s)
        ├── success  → collect partial JSON
        └── timeout  → ray.cancel(force=True)
     
     ReduceActor
        └── merge(partial_results) → final consolidated JSON
```

### Heterogeneous Pipeline Flow

```
Student Query
     │
[SupervisorActor]        ← Gemma 4 26B-MoE  (planning)
     │  Decomposes query → TaskManifest
     │
     ├── [CodeAgent]      Qwen3-Coder-8B   on SGLang
     ├── [GuidanceAgent]  Gemma 4 E4B      on Ollama
     └── [VideoAgent]     Llama3.2-Vision  on llama.cpp
              │
         (timeout / cancel per agent)
              │
     [SynthesizerAgent]   Gemma 4 26B-MoE  on vLLM
              │
      Final Student Response
```

---

## Project Structure

```
.
├── src/
│   ├── benchmark.py              # Concurrent benchmark runner
│   ├── ray_skeleton.py           # Ray Supervisor → SubAgent → Reduce
│   ├── heterogeneous_agents.py   # Full heterogeneous multi-agent pipeline
│   ├── agent_profiles.py         # Model registry, profiles, routing table
│   └── unified_client.py         # UnifiedAgentClient over all 4 engines
├── setup/
│   └── start_engines.sh          # Launch & health-check all 4 engines
├── docker-compose.yml            # Engine cluster definition
├── requirements.txt
└── README.md
```

---

## Prerequisites

| Requirement | Notes |
|---|---|
| Python 3.10+ | |
| Docker & Docker Compose | For running inference engines |
| NVIDIA GPU + `nvidia-container-toolkit` | Required for vLLM and SGLang; Ollama and llama.cpp run on CPU |
| HuggingFace token (optional) | Only needed for gated models (Llama 3 family) |

---

## Installation

**1. Clone the repository**

```bash
git clone <repo-url>
cd <repo-dir>
```

**2. Install Python dependencies**

```bash
pip install -r requirements.txt
```

Key packages: `ray`, `openai>=1.30`, `asyncio`

**3. (Optional) Set your HuggingFace token**

```bash
export HF_TOKEN=hf_your_token_here
# Or add it to a .env file in the project root
```

---

## Quickstart

### Step 1 — Start inference engines

```bash
# GPU — all 4 engines (Ollama, llama.cpp, vLLM, SGLang)
bash setup/start_engines.sh

# CPU only — Ollama + llama.cpp only
bash setup/start_engines.sh --cpu-only
```

The script starts all containers, waits for health checks, and pulls the default Ollama model (`llama3.2:3b`).

### Step 2 — Run the benchmark

```bash
# All engines, 15 concurrent requests, 2 runs each
python src/benchmark.py

# Quick test — Ollama only, 5 concurrent, 1 run
python src/benchmark.py --engines ollama --concurrency 5 --runs 1

# Custom: vLLM + SGLang only, 20 concurrent, 3 runs
python src/benchmark.py --engines vllm sglang --concurrency 20 --runs 3
```

### Step 3 — Run the Ray skeleton

```bash
python src/ray_skeleton.py
```

This runs a 3-agent pipeline where the last agent intentionally times out, demonstrating `ray.cancel()`.

### Step 4 — Run the heterogeneous pipeline

```bash
# Medium complexity (default)
python src/heterogeneous_agents.py

# High complexity — activates stronger model routing
python src/heterogeneous_agents.py --complexity high

# Dry-run — print profile selection table without starting Ray or calling engines
python src/heterogeneous_agents.py --dry-run
```

### Step 5 — Stop all engines

```bash
bash setup/start_engines.sh --stop
```

---

## Inference Engines

All four engines expose an **OpenAI-compatible REST API**, so `UnifiedAgentClient` works identically across all of them.

| Engine | Port | Key Feature | Best For |
|---|---|---|---|
| **Ollama** | 11434 | Pull models on demand, easy CPU/GPU setup | Development, prototyping |
| **llama.cpp** | 8080 | GGUF quantisation, very low memory, continuous batching | CPU-only or low-VRAM GPU |
| **vLLM** | 8000 | PagedAttention, best GPU throughput, prefix caching | Production GPU workloads |
| **SGLang** | 30000 | RadixAttention, best prefix cache hit rate | Multi-agent / shared-context workloads |

### Enabling prefix caching

Prefix caching is what makes vLLM and SGLang dramatically faster on shared-context benchmarks. Both are enabled in `docker-compose.yml`:

```yaml
# vLLM
--enable-prefix-caching

# SGLang (remove --disable-radix-cache flag to enable RadixAttention)
--disable-radix-cache   # Remove or set to false to enable
```

---

## Benchmark Suite

### What it measures

The benchmark replicates the dominant multi-agent pattern: one large shared system prompt (~2500 tokens of tool schemas + domain context) sent with N concurrent short user questions.

| Metric | Description |
|---|---|
| **TTFT** | Time To First Token — how quickly the engine starts responding |
| **TPOT** | Time Per Output Token — generation throughput per token |
| **Total latency** | Wall-clock from request dispatch to last token |
| **RPS** | Successful requests per second (wall-clock based) |

Results are reported at **p50 and p95 percentiles** to expose tail latency variance caused by cache misses.

### CLI options

```
--engines   Which engines to benchmark (default: all). Choices: ollama llama_cpp vllm sglang
--concurrency  Number of simultaneous requests (default: 15)
--runs      Number of benchmark runs to average (default: 2)
--output    Output JSON file path (default: benchmark_results.json)
```

### Shared system prompt

`build_shared_system_prompt()` generates a realistic ~2500-token prompt containing:

- Agent identity and role description
- Four tool schemas (JSON): `search_knowledge_base`, `execute_code`, `retrieve_student_context`, `submit_hint`
- Domain context (Graduate Quantum Information Science course)
- Response format rules

This intentionally stresses prefix-caching mechanisms — engines that can reuse the KV-cache for this prefix will show significantly lower TTFT from the second request onward.

---

## Ray Multi-Agent Pipeline

`ray_skeleton.py` demonstrates the core distributed agent pattern:

### Pipeline phases

**1. Partition** — `ALL_STUDENT_QUESTIONS` (12 questions) split into N chunks, one per sub-agent.

**2. Object Store** — Shared context is placed in Ray's object store via `ray.put()`, enabling zero-copy sharing across all workers.

**3. Map** — All `SubAgentTask` actors are dispatched concurrently. Each calls the inference engine with its assigned question batch.

**4. Timeout/Cancel** — The Supervisor collects results using `asyncio.wait_for()`. Agents that exceed the deadline are cancelled with `ray.cancel(force=True)`. The last agent is intentionally slowed (15s sleep) to demonstrate this.

**5. Reduce** — `ReduceActor` merges all successful partial results into a single consolidated JSON output.

### Key implementation notes

```python
# Zero-copy context sharing
ctx_ref = ray.put(SHARED_CONTEXT)

# Concurrent dispatch
futures = {i: agent.run.remote(chunks[i]) for i, agent in enumerate(agents)}

# Timeout + cancel
result = await asyncio.wait_for(
    asyncio.shield(future),
    timeout=remaining,
)
# On TimeoutError:
ray.cancel(future, force=True)
```

---

## Heterogeneous Agent Pipeline

`heterogeneous_agents.py` extends the skeleton with **role-specific model and engine selection**, producing a complete intelligent tutoring system (ITAS) pipeline.

### Agent roles

| Role | Responsibility | Default Model | Default Engine |
|---|---|---|---|
| `ORCHESTRATOR` | Decompose task, select agents | Gemma 4 26B-MoE | vLLM |
| `CODE` | Bug detection, code analysis | Qwen3-Coder-8B | SGLang |
| `GUIDANCE` | Socratic hints, tutoring | Gemma 4 E4B | Ollama |
| `VIDEO` | Lecture transcript analysis | Llama 3.2-11B-Vision | llama.cpp |
| `SYNTHESIZER` | Merge specialist outputs | Gemma 4 26B-MoE | vLLM |
| `GENERAL` | Fallback | Llama 3.2-3B | Ollama |

### TaskManifest

The Supervisor creates a `TaskManifest` describing the query and which agents are needed:

```python
manifest = TaskManifest(
    original_query=query,
    complexity=TaskComplexity.MEDIUM,
    code_context="...",        # Triggers CodeAgent
    lecture_segment="...",     # Triggers VideoAgent
    # GuidanceAgent is always active
)
```

---

## Agent Profiles & Model Registry

`agent_profiles.py` defines three orthogonal dimensions for every agent:

```
ROLE    × MODEL    × ENGINE
  ↓         ↓          ↓
what      which      where
```

### Model Registry

Models are registered by slug with engine-specific IDs:

```python
MODEL_REGISTRY = {
    "gemma4-26b-moe": {
        "ollama":    "gemma4:26b",
        "vllm":      "unsloth/gemma-4-26B-A4B-it-GGUF",
        "llama_cpp": "gemma-4-26B-A4B-it-Q4_K_M.gguf",
        "sglang":    "unsloth/gemma-4-26B-A4B-it-GGUF",
    },
    "qwen3-coder-8b": { ... },
    "llama32-11b-vision": { ... },
    ...
}
```

### Adding a new model

1. Pull the model: `ollama pull <name>` or add the HuggingFace repo to vLLM
2. Register it in `MODEL_REGISTRY` with the appropriate engine keys
3. Reference it by slug in `PROFILE_CATALOG`

### Complexity-based routing

The `ROUTING_TABLE` maps `(AgentRole, TaskComplexity)` to a profile key:

```python
ROUTING_TABLE = {
    (AgentRole.CODE, TaskComplexity.LOW):    "code-gemma4-e4b",   # Small, fast
    (AgentRole.CODE, TaskComplexity.HIGH):   "code-qwen3coder",   # Specialist
    (AgentRole.ORCHESTRATOR, TaskComplexity.HIGH): "orchestrator-llama33",  # 70B
    ...
}
```

Use `select_profile(role, complexity)` at runtime to get the best available profile.

---

## Configuration

### Engine endpoints (`unified_client.py`)

```python
ENGINES = {
    "ollama":    EngineConfig(base_url="http://localhost:11434/v1", model="llama3.2:3b"),
    "llama_cpp": EngineConfig(base_url="http://localhost:8080/v1",  model="llama-3.2-3b-instruct-q4_k_m"),
    "vllm":      EngineConfig(base_url="http://localhost:8000/v1",  model="llama-3.2-3b"),
    "sglang":    EngineConfig(base_url="http://localhost:30000/v1", model="llama-3.2-3b"),
}
```

### Docker Compose model overrides

Edit `docker-compose.yml` to change the model each engine serves. For llama.cpp, place `.gguf` files in `./models/` and update the `--model` flag. For vLLM/SGLang, update the `--model` argument with the HuggingFace repo ID.

---

## Expected Results

```
════════════════════════════════════════════════════════════════════════════════════════════════════════════════
  MULTI-AGENT CONTEXT BENCHMARK — RESULTS
════════════════════════════════════════════════════════════════════════════════════════════════════════════════
  Engine         Concur   OK  Err   TTFT p50   TTFT p95   TPOT p50  Total p50  Total p95     RPS
  ────────────────────────────────────────────────────────────────────────────────────────────────────────────
  Ollama             15   15    0     1240ms     2100ms     45.2ms     4800ms     7200ms    3.1
  llama.cpp          15   15    0      980ms     1800ms     38.7ms     4200ms     6100ms    3.6
  vLLM               15   15    0      320ms      480ms     12.1ms     1900ms     2800ms    7.9
  SGLang             15   15    0      280ms      410ms     11.4ms     1750ms     2600ms    8.6
  ────────────────────────────────────────────────────────────────────────────────────────────────────────────
```

**Key observation:** vLLM and SGLang show dramatically lower TTFT (~280–320ms vs ~1000–1240ms) because they reuse the KV-cache for the shared system prompt via PagedAttention and RadixAttention respectively. The narrow TTFT p50–p95 gap confirms consistent cache hits.

---

## Key Concepts

**TTFT (Time To First Token)** — The most important latency metric for interactive multi-agent systems. Low TTFT means the orchestrator can start processing sub-agent responses sooner, reducing overall pipeline latency.

**Prefix Caching / KV-Cache Reuse** — When multiple requests share the same prefix (e.g. a system prompt), engines like vLLM and SGLang can compute the key-value attention matrices for that prefix once and reuse them for all subsequent requests. This is the primary reason vLLM/SGLang outperform Ollama/llama.cpp in shared-context scenarios.

**PagedAttention (vLLM)** — Manages KV-cache memory in non-contiguous pages, enabling efficient memory sharing between requests with a common prefix.

**RadixAttention (SGLang)** — Extends prefix caching using a radix tree structure that finds the longest common prefix across all cached sequences, maximising cache hit rates in multi-agent workloads.

**Ray Object Store** — `ray.put()` places an object into Ray's distributed shared memory. All workers receive a reference (`ObjectRef`) and can dereference it without copying data across process boundaries.

---

## Troubleshooting

**Engine not reachable**

```bash
# Check container status
docker compose ps

# Check container logs
docker compose logs vllm --tail 50

# Test endpoint manually
curl http://localhost:8000/health
```

**vLLM / SGLang OOM on startup**

Reduce `--gpu-memory-utilization` in `docker-compose.yml` (default 0.85). For smaller GPUs, switch to a quantised model or use the `--cpu-only` flag to run Ollama + llama.cpp instead.

**Ollama model not found**

```bash
docker exec bench_ollama ollama pull llama3.2:3b
```

**Ray actors not cleaning up**

```bash
ray stop
# Then restart:
python src/ray_skeleton.py
```

**`resolve_model` KeyError**

The requested `(model_slug, engine_key)` combination is not registered in `MODEL_REGISTRY`. Check `agent_profiles.py` and add the missing entry, or use a different engine key that the model supports.