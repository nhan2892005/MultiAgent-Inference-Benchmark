#!/usr/bin/env bash
# =============================================================================
# start_engines.sh  —  Launch & verify all 4 inference engines
# Usage:  bash start_engines.sh [--cpu-only] [--stop]
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_NAME="llama3.2:3b"            # Ollama model tag
ENGINES=("ollama:11434" "llama_cpp:8080" "vllm:8000" "sglang:30000")
CPU_ONLY=false
STOP=false

# ── Parse flags ──────────────────────────────────────────────────────────────
for arg in "$@"; do
  case $arg in
    --cpu-only) CPU_ONLY=true ;;
    --stop)     STOP=true ;;
  esac
done

# ── Stop mode ────────────────────────────────────────────────────────────────
if $STOP; then
  echo "🛑  Stopping all engines..."
  docker compose -f "$SCRIPT_DIR/docker-compose.yml" down
  echo "✅  All engines stopped."
  exit 0
fi

# ── GPU check ────────────────────────────────────────────────────────────────
if $CPU_ONLY; then
  echo "⚠️   CPU-only mode: vLLM and SGLang require GPU and will be skipped."
  echo "    Only Ollama and llama.cpp will start."
  COMPOSE_PROFILES="cpu"
  ENGINES=("ollama:11434" "llama_cpp:8080")
else
  COMPOSE_PROFILES="all"
fi

# ── Start containers ─────────────────────────────────────────────────────────
echo "🚀  Starting inference engine cluster..."
cd "$SCRIPT_DIR"
docker compose up -d 2>&1 | tail -5

# ── Wait for health checks ───────────────────────────────────────────────────
echo ""
echo "⏳  Waiting for engines to become healthy..."

wait_for_engine() {
  local name=$1
  local port=$2
  local endpoint="${3:-/health}"
  local max_retries=30
  local delay=5

  for i in $(seq 1 $max_retries); do
    if curl -sf "http://localhost:${port}${endpoint}" > /dev/null 2>&1; then
      printf "  ✅  %-12s ready on :%s\n" "$name" "$port"
      return 0
    fi
    printf "  ⏳  %-12s not ready yet (attempt %d/%d)...\r" "$name" "$i" "$max_retries"
    sleep $delay
  done

  printf "  ❌  %-12s FAILED to start on :%s\n" "$name" "$port"
  return 1
}

wait_for_engine "Ollama"    11434 "/api/tags"
wait_for_engine "llama.cpp" 8080  "/health"

if ! $CPU_ONLY; then
  wait_for_engine "vLLM"    8000  "/health"
  wait_for_engine "SGLang"  30000 "/health"
fi

# ── Pull Ollama model ─────────────────────────────────────────────────────────
echo ""
echo "📦  Pulling Ollama model: ${MODEL_NAME}..."
docker exec bench_ollama ollama pull "$MODEL_NAME"
echo "  ✅  Model ready."

# ── Summary table ─────────────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║            Inference Engine Cluster — Ready                 ║"
echo "╠═══════════════╦══════════╦════════════════════════════════════╣"
echo "║ Engine        ║ Port     ║ OpenAI Base URL                    ║"
echo "╠═══════════════╬══════════╬════════════════════════════════════╣"
echo "║ Ollama        ║ 11434    ║ http://localhost:11434/v1          ║"
echo "║ llama.cpp     ║ 8080     ║ http://localhost:8080/v1           ║"
if ! $CPU_ONLY; then
  echo "║ vLLM          ║ 8000     ║ http://localhost:8000/v1           ║"
  echo "║ SGLang        ║ 30000    ║ http://localhost:30000/v1          ║"
fi
echo "╚═══════════════╩══════════╩════════════════════════════════════╝"
echo ""
echo "Run benchmark:  python src/benchmark.py"
echo "Stop engines:   bash setup/start_engines.sh --stop"