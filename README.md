# Multi-Agent Inference Benchmark & Ray Skeleton

A benchmark harness + distributed agent skeleton for evaluating local LLM
inference engines under **shared-context, high-concurrency** workloads.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Scope 1 & 2                                 │
│                                                                 │
│  UnifiedAgentClient  →  OpenAI-compatible API                   │
│                              │                                  │
│          ┌───────────────────┼───────────────────┐              │
│          ▼           ▼       ▼         ▼         │              │
│       Ollama    llama.cpp   vLLM    SGLang       │              │
│       :11434    :8080       :8000   :30000       │              │
│                                                  │              │
│  benchmark.py                                    │              │
│   ├── build_shared_system_prompt() ~2500 tokens  │              │
│   ├── 15 concurrent requests (one per question)  │              │
│   ├── Measure: TTFT, TPOT, Total Latency         │              │
│   └── Print results table + save JSON            │              │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                     Scope 3                                     │
│                                                                 │
│  SupervisorActor (Main Agent)                                   │
│       │                                                         │
│       ├─── ray.put(shared_context)  → Object Store              │
│       │                                                         │
│       ├─── SubAgentTask-0  [questions 0-3]       fast           │
│       ├─── SubAgentTask-1  [questions 4-7]       fast           │
│       └─── SubAgentTask-2  [questions 8-11]      SLOW→CANCEL    │
│                                                                 │
│       asyncio.wait_for(future, timeout=8s)                      │
│          ├── success  → collect partial JSON                    │
│          └── timeout  → ray.cancel(force=True)                  │
│                                                                 │
│       ReduceActor                                               │
│          └── merge(partial_results) → final consolidated JSON   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Quickstart

### 1 — Install Python dependencies
```bash
pip install -r requirements.txt
```

### 2 — Start inference engines
```bash
# GPU (all 4 engines)
bash setup/start_engines.sh

# CPU only (Ollama + llama.cpp)
bash setup/start_engines.sh --cpu-only
```

### 3 — Run benchmark
```bash
# All engines, 15 concurrent requests, 2 runs each
python src/benchmark.py

# Quick test — just Ollama, 5 concurrent
python src/benchmark.py --engines ollama --concurrency 5 --runs 1
```

### 4 — Run Ray multi-agent skeleton
```bash
python src/ray_skeleton.py
```

---

## Engine Endpoints

| Engine    | Port  | Key Feature                        |
|-----------|-------|------------------------------------|
| Ollama    | 11434 | Easy CPU/GPU, pulls models on-demand |
| llama.cpp | 8080  | GGUF quant, low memory, cont. batch |
| vLLM      | 8000  | PagedAttention, best GPU throughput |
| SGLang    | 30000 | RadixAttention, best prefix caching |

---

## Expected Benchmark Output

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

**Key observation**: vLLM/SGLang show dramatically lower TTFT because they
reuse the KV-cache for the shared system prompt (PagedAttention /
RadixAttention). TTFT p50 vs p95 gap is narrow → cache hits are consistent.

---

## Stop engines
```bash
bash setup/start_engines.sh --stop
```