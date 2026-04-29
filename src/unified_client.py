"""
unified_client.py
=================
UnifiedAgentClient — a single interface over any OpenAI-compatible inference
engine (Ollama, llama.cpp, vLLM, SGLang).

Usage:
    from unified_client import UnifiedAgentClient, EngineConfig, ENGINES

    client = UnifiedAgentClient(ENGINES["vllm"])
    response = await client.chat("Explain KV-cache in one sentence.")
    tokens   = await client.stream_chat("...", on_token=print)
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import AsyncIterator, Callable, Optional

import openai                        # pip install openai>=1.30
from openai import AsyncOpenAI


# ── Engine registry ───────────────────────────────────────────────────────────

@dataclass(frozen=True)
class EngineConfig:
    """Immutable description of a single inference engine endpoint."""
    name: str
    base_url: str
    api_key: str        = "not-needed"   # Most local servers ignore this
    model: str          = "llama3.2:3b"  # Override per engine as needed
    timeout: float      = 120.0          # Request-level timeout (seconds)
    max_tokens: int     = 512
    temperature: float  = 0.0            # Deterministic for benchmarks


ENGINES: dict[str, EngineConfig] = {
    "ollama": EngineConfig(
        name="Ollama",
        base_url="http://localhost:11434/v1",
        model="llama3.2:3b",
    ),
    "llama_cpp": EngineConfig(
        name="llama.cpp",
        base_url="http://localhost:8080/v1",
        model="llama-3.2-3b-instruct-q4_k_m",  # Match --model filename
    ),
    "vllm": EngineConfig(
        name="vLLM",
        base_url="http://localhost:8000/v1",
        model="llama-3.2-3b",                   # --served-model-name in compose
    ),
    "sglang": EngineConfig(
        name="SGLang",
        base_url="http://localhost:30000/v1",
        model="llama-3.2-3b",
    ),
}


# ── Timing record ─────────────────────────────────────────────────────────────

@dataclass
class RequestMetrics:
    engine: str
    request_id: int
    prompt_tokens: int  = 0
    output_tokens: int  = 0
    ttft_ms: float      = 0.0   # Time To First Token (ms)
    tpot_ms: float      = 0.0   # Time Per Output Token (ms)
    total_ms: float     = 0.0   # Wall-clock from dispatch to last token (ms)
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.error is None

    def __str__(self) -> str:
        if not self.ok:
            return f"[{self.engine}] req#{self.request_id} ERROR: {self.error}"
        return (
            f"[{self.engine}] req#{self.request_id} | "
            f"TTFT={self.ttft_ms:6.0f}ms | "
            f"TPOT={self.tpot_ms:5.1f}ms | "
            f"Total={self.total_ms:6.0f}ms | "
            f"Tokens={self.prompt_tokens}+{self.output_tokens}"
        )


# ── Main client ───────────────────────────────────────────────────────────────

class UnifiedAgentClient:
    """
    Drop-in OpenAI client wrapper for any local inference engine.

    Key design choices
    ------------------
    - Single AsyncOpenAI instance per client (connection pool reuse).
    - All public methods are async to play nicely with asyncio concurrency.
    - stream_chat() returns both the full text AND precise token timing.
    - No engine-specific logic lives here — everything is config-driven.
    """

    def __init__(self, config: EngineConfig):
        self.cfg = config
        self._client = AsyncOpenAI(
            base_url=config.base_url,
            api_key=config.api_key,
            timeout=openai.Timeout(connect=10.0, read=config.timeout, write=10.0, pool=5.0),
            max_retries=0,  # Caller controls retry semantics
        )

    # ── Non-streaming chat ────────────────────────────────────────────────────

    async def chat(
        self,
        user_prompt: str,
        system_prompt: str = "",
        request_id: int = 0,
    ) -> tuple[str, RequestMetrics]:
        """
        Single-turn chat (no streaming).  Cheaper on TTFT measurement because
        the OpenAI SDK buffers everything, so we use wall-clock only.
        """
        messages = _build_messages(system_prompt, user_prompt)
        metrics = RequestMetrics(engine=self.cfg.name, request_id=request_id)

        t0 = time.perf_counter()
        try:
            resp = await self._client.chat.completions.create(
                model=self.cfg.model,
                messages=messages,
                max_tokens=self.cfg.max_tokens,
                temperature=self.cfg.temperature,
                stream=False,
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000
            content = resp.choices[0].message.content or ""

            # Usage stats (may be None on some engines — guard carefully)
            usage = resp.usage
            metrics.prompt_tokens = usage.prompt_tokens  if usage else 0
            metrics.output_tokens = usage.completion_tokens if usage else len(content.split())
            metrics.total_ms      = elapsed_ms
            metrics.ttft_ms       = elapsed_ms          # Approximation (non-stream)
            metrics.tpot_ms       = (
                elapsed_ms / max(metrics.output_tokens, 1)
            )
            return content, metrics

        except Exception as exc:
            metrics.error   = str(exc)
            metrics.total_ms = (time.perf_counter() - t0) * 1000
            return "", metrics

    # ── Streaming chat (accurate TTFT) ────────────────────────────────────────

    async def stream_chat(
        self,
        user_prompt: str,
        system_prompt: str = "",
        request_id: int = 0,
        on_token: Optional[Callable[[str], None]] = None,
    ) -> tuple[str, RequestMetrics]:
        """
        Streaming chat with precise TTFT measurement.

        t0           = request dispatched
        first_token  = first non-empty chunk received  → TTFT
        last_token   = final chunk                     → Total latency
        TPOT         = (Total - TTFT) / (output_tokens - 1)
        """
        messages = _build_messages(system_prompt, user_prompt)
        metrics  = RequestMetrics(engine=self.cfg.name, request_id=request_id)
        chunks: list[str] = []
        first_token_time: Optional[float] = None

        t0 = time.perf_counter()
        try:
            stream = await self._client.chat.completions.create(
                model=self.cfg.model,
                messages=messages,
                max_tokens=self.cfg.max_tokens,
                temperature=self.cfg.temperature,
                stream=True,
                stream_options={"include_usage": True},  # Works on vLLM/SGLang
            )

            async for chunk in stream:
                # Usage block arrives in the final sentinel chunk
                if chunk.usage:
                    metrics.prompt_tokens = chunk.usage.prompt_tokens
                    metrics.output_tokens = chunk.usage.completion_tokens

                delta = chunk.choices[0].delta.content if chunk.choices else None
                if not delta:
                    continue

                now = time.perf_counter()
                if first_token_time is None:
                    first_token_time = now               # Record TTFT moment

                chunks.append(delta)
                if on_token:
                    on_token(delta)

            t_end = time.perf_counter()
            full_text = "".join(chunks)

            # ── Derive metrics ────────────────────────────────────────────────
            if metrics.output_tokens == 0:               # Engine didn't report
                metrics.output_tokens = len(full_text.split())

            metrics.total_ms = (t_end - t0) * 1000
            metrics.ttft_ms  = ((first_token_time or t_end) - t0) * 1000
            generation_ms    = metrics.total_ms - metrics.ttft_ms
            metrics.tpot_ms  = generation_ms / max(metrics.output_tokens - 1, 1)

            return full_text, metrics

        except Exception as exc:
            metrics.error    = str(exc)
            metrics.total_ms = (time.perf_counter() - t0) * 1000
            return "", metrics

    # ── Utility ───────────────────────────────────────────────────────────────

    async def is_alive(self) -> bool:
        """Quick health-check — tries to list models."""
        try:
            await self._client.models.list()
            return True
        except Exception:
            return False

    async def close(self) -> None:
        await self._client.close()

    def __repr__(self) -> str:
        return f"UnifiedAgentClient(engine={self.cfg.name!r}, url={self.cfg.base_url!r})"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_messages(system_prompt: str, user_prompt: str) -> list[dict]:
    messages: list[dict] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})
    return messages