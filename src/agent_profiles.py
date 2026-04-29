"""
agent_profiles.py
=================
Defines the "who + where + how" of every agent in the system.

Three orthogonal dimensions
---------------------------
  1. ROLE       — what job this agent does (orchestrator, code, guidance, …)
  2. MODEL      — which LLM weights (Gemma 4, Qwen3, Llama 3.2, …)
  3. ENGINE     — which inference runtime serves it (Ollama, vLLM, SGLang, llama.cpp)

Any combination is valid.  Examples from the Unsloth collection provided:
  • Main agent    → Gemma 4 26B-A4B  on vLLM      (big, smart, GPU)
  • Code agent    → Qwen3-Coder 8B   on SGLang     (fast, code-specialist)
  • Guidance agent→ Gemma 4 E4B      on Ollama     (small, CPU-friendly)
  • Video agent   → Llama 3.2-11B-V  on llama.cpp  (vision-capable)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from unified_client import EngineConfig, ENGINES


# =============================================================================
# Agent Role taxonomy
# =============================================================================

class AgentRole(str, Enum):
    """
    High-level functional role.  The Supervisor uses this to decide
    which AgentProfile to assign to each sub-task.
    """
    ORCHESTRATOR = "orchestrator"   # Main / Supervisor agent
    CODE         = "code"           # Code inspection, debugging
    GUIDANCE     = "guidance"       # Socratic hints, pedagogy
    VIDEO        = "video"          # Lecture transcript analysis
    SYNTHESIZER  = "synthesizer"    # Merge parallel agent outputs
    GENERAL      = "general"        # Fallback / no specialisation


# =============================================================================
# Task complexity hint (used by Supervisor for model routing)
# =============================================================================

class TaskComplexity(str, Enum):
    LOW    = "low"      # Short factual, keyword lookup
    MEDIUM = "medium"   # Multi-step reasoning, short code
    HIGH   = "high"     # Long derivation, multi-file code, synthesis


# =============================================================================
# AgentProfile — the complete identity of one agent instance
# =============================================================================

@dataclass
class AgentProfile:
    """
    Describes everything needed to spin up one agent actor.

    Fields
    ------
    role          : Functional role (drives prompt template selection)
    engine_key    : Key into ENGINES dict  (e.g. "vllm", "ollama")
    model_id      : Model name/tag for the chosen engine
    display_name  : Human-readable label for logs/dashboards
    system_prompt : Role-specific system prompt injected at the actor level
    max_tokens    : Output budget (smaller roles → smaller budget)
    temperature   : 0.0 for deterministic sub-agents; higher for creative roles
    extra_params  : Engine-specific extras (e.g. {"top_p": 0.95})
    """
    role:         AgentRole
    engine_key:   str
    model_id:     str
    display_name: str             = ""
    system_prompt: str            = ""
    max_tokens:   int             = 512
    temperature:  float           = 0.0
    extra_params: dict            = field(default_factory=dict)

    def __post_init__(self):
        if not self.display_name:
            self.display_name = f"{self.role.value}/{self.model_id.split('/')[-1]}"

    @property
    def engine_config(self) -> EngineConfig:
        """Build a ready-to-use EngineConfig from the registered engine."""
        base = ENGINES[self.engine_key]
        return EngineConfig(
            name=self.display_name,
            base_url=base.base_url,
            api_key=base.api_key,
            model=self.model_id,
            timeout=base.timeout,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )


# =============================================================================
# Model Registry — available models per engine
# =============================================================================
#
# Key  : arbitrary slug used in profile definitions
# Value: (engine_key, model_id_for_that_engine)
#
# ── HOW TO ADD A NEW MODEL ────────────────────────────────────────────────────
#  1. Pull the GGUF with Ollama:      ollama pull <name>
#  2. Or add the HF repo to vLLM:    --model <hf_id>
#  3. Register below → reference by slug in PROFILE_CATALOG
# ─────────────────────────────────────────────────────────────────────────────

MODEL_REGISTRY: dict[str, dict[str, str]] = {
    # ── Gemma 4 family (Google via Unsloth quants) ────────────────────────────
    "gemma4-26b-moe": {           # MoE: 26B total / 4B active — best quality/cost
        "ollama":    "gemma4:26b",
        "vllm":      "unsloth/gemma-4-26B-A4B-it-GGUF",
        "llama_cpp": "gemma-4-26B-A4B-it-Q4_K_M.gguf",
        "sglang":    "unsloth/gemma-4-26B-A4B-it-GGUF",
    },
    "gemma4-31b": {               # Dense 31B — highest quality, needs big GPU
        "ollama":    "gemma4:31b",
        "vllm":      "unsloth/gemma-4-31B-it-GGUF",
        "llama_cpp": "gemma-4-31B-it-Q4_K_M.gguf",
    },
    "gemma4-e4b": {               # Edge 4B — fast, fits on CPU/small GPU
        "ollama":    "gemma4:4b",
        "vllm":      "unsloth/gemma-4-E4B-it-GGUF",
        "llama_cpp": "gemma-4-E4B-it-Q4_K_M.gguf",
        "sglang":    "unsloth/gemma-4-E4B-it-GGUF",
    },
    "gemma4-e2b": {               # Micro 2B — CPU-only, very low latency
        "ollama":    "gemma4:2b",
        "llama_cpp": "gemma-4-E2B-it-Q4_K_M.gguf",
    },

    # ── Qwen3 family (Alibaba) ────────────────────────────────────────────────
    "qwen3-coder-8b": {           # Code specialist — best for CODE role
        "ollama":    "qwen2.5-coder:7b",
        "vllm":      "Qwen/Qwen2.5-Coder-7B-Instruct",
        "sglang":    "Qwen/Qwen2.5-Coder-7B-Instruct",
    },
    "qwen3-30b-moe": {            # Qwen3 30B MoE — strong reasoning
        "ollama":    "qwen3:30b",
        "vllm":      "Qwen/Qwen3-30B-A3B",
        "sglang":    "Qwen/Qwen3-30B-A3B",
    },

    # ── Llama 3.x family (Meta) ───────────────────────────────────────────────
    "llama32-3b": {               # Tiny, fast baseline
        "ollama":    "llama3.2:3b",
        "vllm":      "meta-llama/Llama-3.2-3B-Instruct",
        "sglang":    "meta-llama/Llama-3.2-3B-Instruct",
        "llama_cpp": "llama-3.2-3b-instruct-q4_k_m.gguf",
    },
    "llama32-11b-vision": {       # Multimodal — good for VIDEO role
        "ollama":    "llama3.2-vision:11b",
        "vllm":      "meta-llama/Llama-3.2-11B-Vision-Instruct",
        "llama_cpp": "llama-3.2-11b-vision-instruct-q4_k_m.gguf",
    },
    "llama33-70b": {              # Largest Llama — orchestrator-grade
        "vllm":      "meta-llama/Llama-3.3-70B-Instruct",
        "sglang":    "meta-llama/Llama-3.3-70B-Instruct",
    },
}


def resolve_model(slug: str, engine_key: str) -> str:
    """
    Look up the engine-specific model ID for a given model slug.
    Raises KeyError with a helpful message if the combination isn't registered.
    """
    if slug not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model slug '{slug}'. Available: {list(MODEL_REGISTRY)}")
    engine_map = MODEL_REGISTRY[slug]
    if engine_key not in engine_map:
        available = list(engine_map.keys())
        raise KeyError(
            f"Model '{slug}' not registered for engine '{engine_key}'. "
            f"Available engines for this model: {available}"
        )
    return engine_map[engine_key]


# =============================================================================
# Profile Catalog — named, reusable AgentProfile configurations
# =============================================================================
#
# These are the "presets" the Supervisor picks from.
# You can override any field when instantiating for a specific task.
# ─────────────────────────────────────────────────────────────────────────────

def _make_profile(
    role: AgentRole,
    model_slug: str,
    engine_key: str,
    max_tokens: int = 512,
    temperature: float = 0.0,
    system_prompt: str = "",
) -> AgentProfile:
    return AgentProfile(
        role=role,
        engine_key=engine_key,
        model_id=resolve_model(model_slug, engine_key),
        max_tokens=max_tokens,
        temperature=temperature,
        system_prompt=system_prompt,
    )


# Default system prompts per role
_ROLE_PROMPTS: dict[AgentRole, str] = {
    AgentRole.ORCHESTRATOR: (
        "You are the main orchestration agent. Break down the incoming task, "
        "decide which specialist agents to spawn, and synthesise their outputs."
    ),
    AgentRole.CODE: (
        "You are a code specialist. Analyse code for bugs, suggest fixes, and "
        "explain programming concepts. Return structured JSON with keys: "
        "bugs (list), fixes (list), concepts (list)."
    ),
    AgentRole.GUIDANCE: (
        "You are a Socratic tutoring agent. Guide students to answers through "
        "leading questions. Never reveal the full solution. "
        "Return JSON: hints (list[str]), difficulty (float 1-10)."
    ),
    AgentRole.VIDEO: (
        "You are a lecture-content agent. Connect student questions to the "
        "provided lecture transcript. Return JSON: references (list), "
        "summary (str), key_concepts (list)."
    ),
    AgentRole.SYNTHESIZER: (
        "You are the synthesis agent. Merge outputs from multiple specialist "
        "agents into one coherent, well-structured student response."
    ),
    AgentRole.GENERAL: (
        "You are a general-purpose assistant. Answer questions accurately."
    ),
}


PROFILE_CATALOG: dict[str, AgentProfile] = {

    # ── Orchestrator / Main Agent options ─────────────────────────────────────
    "orchestrator-gemma4-moe": AgentProfile(
        role=AgentRole.ORCHESTRATOR,
        engine_key="vllm",
        model_id=resolve_model("gemma4-26b-moe", "vllm"),
        display_name="Orchestrator/Gemma4-26B-MoE",
        system_prompt=_ROLE_PROMPTS[AgentRole.ORCHESTRATOR],
        max_tokens=1024,
        temperature=0.1,   # Slight creativity for planning
    ),
    "orchestrator-llama33": AgentProfile(
        role=AgentRole.ORCHESTRATOR,
        engine_key="vllm",
        model_id=resolve_model("llama33-70b", "vllm"),
        display_name="Orchestrator/Llama3.3-70B",
        system_prompt=_ROLE_PROMPTS[AgentRole.ORCHESTRATOR],
        max_tokens=1024,
        temperature=0.1,
    ),

    # ── Code sub-agent options ────────────────────────────────────────────────
    "code-qwen3coder": AgentProfile(
        role=AgentRole.CODE,
        engine_key="sglang",          # SGLang RadixAttention best for code tasks
        model_id=resolve_model("qwen3-coder-8b", "sglang"),
        display_name="Code/Qwen3-Coder-8B",
        system_prompt=_ROLE_PROMPTS[AgentRole.CODE],
        max_tokens=512,
        temperature=0.0,
    ),
    "code-gemma4-e4b": AgentProfile(
        role=AgentRole.CODE,
        engine_key="ollama",          # CPU fallback
        model_id=resolve_model("gemma4-e4b", "ollama"),
        display_name="Code/Gemma4-E4B",
        system_prompt=_ROLE_PROMPTS[AgentRole.CODE],
        max_tokens=512,
        temperature=0.0,
    ),

    # ── Guidance sub-agent options ────────────────────────────────────────────
    "guidance-gemma4-e4b": AgentProfile(
        role=AgentRole.GUIDANCE,
        engine_key="ollama",
        model_id=resolve_model("gemma4-e4b", "ollama"),
        display_name="Guidance/Gemma4-E4B",
        system_prompt=_ROLE_PROMPTS[AgentRole.GUIDANCE],
        max_tokens=256,
        temperature=0.0,
    ),
    "guidance-gemma4-e2b": AgentProfile(
        role=AgentRole.GUIDANCE,
        engine_key="llama_cpp",       # Ultra-light, CPU-only
        model_id=resolve_model("gemma4-e2b", "llama_cpp"),
        display_name="Guidance/Gemma4-E2B",
        system_prompt=_ROLE_PROMPTS[AgentRole.GUIDANCE],
        max_tokens=256,
        temperature=0.0,
    ),

    # ── Video sub-agent options ───────────────────────────────────────────────
    "video-llama32-vision": AgentProfile(
        role=AgentRole.VIDEO,
        engine_key="llama_cpp",
        model_id=resolve_model("llama32-11b-vision", "llama_cpp"),
        display_name="Video/Llama3.2-11B-Vision",
        system_prompt=_ROLE_PROMPTS[AgentRole.VIDEO],
        max_tokens=512,
        temperature=0.0,
    ),
    "video-gemma4-26b-moe": AgentProfile(
        role=AgentRole.VIDEO,
        engine_key="vllm",
        model_id=resolve_model("gemma4-26b-moe", "vllm"),
        display_name="Video/Gemma4-26B-MoE",
        system_prompt=_ROLE_PROMPTS[AgentRole.VIDEO],
        max_tokens=512,
        temperature=0.0,
    ),

    # ── Synthesizer options ───────────────────────────────────────────────────
    "synthesizer-gemma4-moe": AgentProfile(
        role=AgentRole.SYNTHESIZER,
        engine_key="vllm",
        model_id=resolve_model("gemma4-26b-moe", "vllm"),
        display_name="Synthesizer/Gemma4-26B-MoE",
        system_prompt=_ROLE_PROMPTS[AgentRole.SYNTHESIZER],
        max_tokens=1024,
        temperature=0.0,
    ),

    # ── Cheap fallbacks (when GPU unavailable) ────────────────────────────────
    "any-llama32-3b": AgentProfile(
        role=AgentRole.GENERAL,
        engine_key="ollama",
        model_id=resolve_model("llama32-3b", "ollama"),
        display_name="General/Llama3.2-3B",
        system_prompt=_ROLE_PROMPTS[AgentRole.GENERAL],
        max_tokens=512,
        temperature=0.0,
    ),
}


# =============================================================================
# Complexity → Profile routing table
# =============================================================================
# Maps (AgentRole, TaskComplexity) → preferred profile key in PROFILE_CATALOG.
# The Supervisor consults this at spawn time.

ROUTING_TABLE: dict[tuple[AgentRole, TaskComplexity], str] = {
    # Orchestrator always uses the strongest available model
    (AgentRole.ORCHESTRATOR, TaskComplexity.LOW):    "orchestrator-gemma4-moe",
    (AgentRole.ORCHESTRATOR, TaskComplexity.MEDIUM): "orchestrator-gemma4-moe",
    (AgentRole.ORCHESTRATOR, TaskComplexity.HIGH):   "orchestrator-llama33",

    # Code agent: Qwen3-Coder for medium/high; tiny Gemma for simple tasks
    (AgentRole.CODE, TaskComplexity.LOW):    "code-gemma4-e4b",
    (AgentRole.CODE, TaskComplexity.MEDIUM): "code-qwen3coder",
    (AgentRole.CODE, TaskComplexity.HIGH):   "code-qwen3coder",

    # Guidance: always small/fast (hints don't need big model)
    (AgentRole.GUIDANCE, TaskComplexity.LOW):    "guidance-gemma4-e2b",
    (AgentRole.GUIDANCE, TaskComplexity.MEDIUM): "guidance-gemma4-e4b",
    (AgentRole.GUIDANCE, TaskComplexity.HIGH):   "guidance-gemma4-e4b",

    # Video: vision model when available; MoE for transcript-only
    (AgentRole.VIDEO, TaskComplexity.LOW):    "video-gemma4-26b-moe",
    (AgentRole.VIDEO, TaskComplexity.MEDIUM): "video-llama32-vision",
    (AgentRole.VIDEO, TaskComplexity.HIGH):   "video-llama32-vision",

    # Synthesizer always uses a capable model
    (AgentRole.SYNTHESIZER, TaskComplexity.LOW):    "synthesizer-gemma4-moe",
    (AgentRole.SYNTHESIZER, TaskComplexity.MEDIUM): "synthesizer-gemma4-moe",
    (AgentRole.SYNTHESIZER, TaskComplexity.HIGH):   "synthesizer-gemma4-moe",
}


def select_profile(
    role: AgentRole,
    complexity: TaskComplexity,
    fallback_key: str = "any-llama32-3b",
) -> AgentProfile:
    """
    Look up the best AgentProfile for a given (role, complexity) pair.
    Falls back to `fallback_key` if the routing table has no entry.
    """
    key = ROUTING_TABLE.get((role, complexity))
    if key and key in PROFILE_CATALOG:
        return PROFILE_CATALOG[key]
    return PROFILE_CATALOG[fallback_key]