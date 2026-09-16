import os

# --- Models ---
# Three ~30B models: 2x MoE (3B active), 1x Dense (27B active)
# All run in non-thinking mode by default (matches Kim et al.)

MODELS = {
    "qwen35b-a3b": {
        "model_path": "Qwen/Qwen3.5-35B-A3B-GPTQ-Int4",
        "architecture": "moe",
        "params_total": "35B",
        "params_active": "3B",
        "quantization": "gptq",
        "tool_call_parser": "qwen3_coder",
        "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
        "sglang_extra_args": ["--trust-remote-code"],
    },
    "glm47-flash": {
        "model_path": "zai-org/GLM-4.7-Flash",
        "architecture": "moe",
        "params_total": "30B",
        "params_active": "3B",
        "quantization": None,  # FP8 or BF16; no official GPTQ-Int4
        "tool_call_parser": "glm47",
        "extra_body": None,
        "sglang_extra_args": ["--attention-backend", "triton", "--trust-remote-code"],
    },
    "mixtral-8x7b-fp8": {
        # Mixtral 8x7B Instruct FP8 — Mistral MoE, mature SGLang support since 0.4.
        # 46.7 B total, 12.9 B active (much higher activation ratio than Qwen 3B-active
        # → distinct MoE profile). ~24 GB weights at FP8, leaves ~24 GB for KV pool.
        # Fallback candidate if GLM/Gemma init issues persist.
        "model_path": "RedHatAI/Mixtral-8x7B-Instruct-v0.1-FP8",
        "architecture": "moe",
        "params_total": "46.7B",
        "params_active": "12.9B",
        "quantization": "fp8",
        "tool_call_parser": "mistral",
        "extra_body": None,
        "sglang_extra_args": ["--trust-remote-code"],
    },
    "glm47-flash-fp8": {
        # Community FP8 quant of GLM-4.7-Flash for 48 GB A6000 fit.
        # Base BF16 (62 GB weights) OOMs on A6000 even at mem-fraction 0.70.
        # FP8 shrinks weights to ~30 GB, leaves ~15 GB for KV + MoE expert alloc.
        "model_path": "Geodd/GLM-4.7-Flash-FP8",
        "architecture": "moe",
        "params_total": "30B",
        "params_active": "3B",
        "quantization": "fp8",
        "tool_call_parser": "glm47",
        "extra_body": None,
        "sglang_extra_args": ["--attention-backend", "triton", "--trust-remote-code"],
    },
    "qwen35-27b": {
        "model_path": "Qwen/Qwen3.5-27B-GPTQ-Int4",
        "architecture": "dense",
        "params_total": "27B",
        "params_active": "27B",
        "quantization": "gptq",
        "tool_call_parser": "qwen3_coder",
        "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
        "sglang_extra_args": ["--trust-remote-code"],
    },
    "qwen35-27b-fp8": {
        "model_path": "Qwen/Qwen3.5-27B-FP8",
        "architecture": "dense",  # hybrid attention (Gated DeltaNet + Gated Attention), not MoE
        "params_total": "27B",
        "params_active": "27B",
        "quantization": "fp8",  # E4M3, fine-grained block_size=128
        # DIAGNOSIS (revised — my earlier v2/v3 chases were wrong-direction):
        #   Qwen 3.5-27B emits the Qwen3-Coder XML tool-call format natively:
        #     <tool_call><function=bash><parameter=command>pwd</parameter>...</tool_call>
        #   (Confirmed at github.com/QwenLM/Qwen3.6/issues/125.)
        #   SGLang's qwen3_coder parser IS the correct match for this format.
        #
        # v1 (qwen3_coder + no /testbed warning): 0/10 — model stuck searching
        #   nonexistent /testbed via `find`, exhausted 10 react steps, empty answer.
        #   PARSER WAS FINE — problem was the prompt.
        # v2/v3: I misdiagnosed v1 as parser issue and changed to qwen/qwen25.
        #   Both fail to parse XML format → empty tool_calls[] → model quits.
        # v4: revert to qwen3_coder (correct parser) with the /testbed warning in
        #   prompt (already in benchmarks_swebench.py:609). This combination was
        #   never tested before.
        "tool_call_parser": "qwen3_coder",
        "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
        "sglang_extra_args": ["--trust-remote-code"],
    },
    "qwen35b-a3b-fp8": {
        # Qwen 3.5-35B-A3B with Qwen's OFFICIAL fine-grained FP8 (block_size=128),
        # near-lossless per model card. The MoE arm the study originally planned:
        # 35B total / 3B active, same family + tokenizer + FP8 quant as the 27B
        # dense arm → cleanest possible MoE-vs-dense architecture isolation.
        # The old "qwen35b-a3b" entry above used GPTQ-Int4, ruled out after the
        # A4000 study showed a 9-17pp INT4 accuracy hit; the official FP8
        # checkpoint was never revisited until 2026-07-14 (see project_log).
        # VRAM: ~35 GB weights on 48 GB A6000. KV is tiny — hybrid attention:
        # only 10/40 layers are standard attention, with 2 KV heads (rest is
        # Gated DeltaNet, O(1) state) — so a ~5-7 GB KV pool suffices.
        "model_path": "Qwen/Qwen3.5-35B-A3B-FP8",
        "architecture": "moe",  # 256 experts, 8 routed + 1 shared active per token
        "params_total": "35B",
        "params_active": "3B",
        "quantization": "fp8",  # E4M3, fine-grained block_size=128, official
        # Same native Qwen3-Coder XML tool-call format as the 27B (see the
        # qwen35-27b-fp8 diagnosis below) → qwen3_coder parser.
        "tool_call_parser": "qwen3_coder",
        "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
        "sglang_extra_args": ["--trust-remote-code"],
    },
    "gemma4-31b-qat-w4a16": {
        # Gemma 4-31B DENSE with Google's official QAT w4a16 quantization.
        # Replaces the retracted gemma4-26b-a4b-fp8 MoE candidate — see project_log
        # 2026-07-08 entry. Dense architecture avoids SGLang's SiLU-only MoE
        # assertion. QAT quantization outperforms community AWQ per Google's docs
        # (weights adjusted during training vs post-hoc). ~21 GB weights on 48 GB
        # A6000, leaves ~27 GB for KV cache.
        "model_path": "google/gemma-4-31B-it-qat-w4a16",
        "architecture": "dense",
        "params_total": "31B",
        "params_active": "31B",
        "quantization": "compressed-tensors",  # w4a16 QAT
        # Updated 2026-07-08 after pilot 16106750 diagnosis: SGLang 0.5.14 has
        # native gemma4 tool_call_parser (auto-detected from model chat template).
        # Prior 'pythonic' setting was leftover from 0.5.10 and caused n=1/10
        # tasks in the pilot — pythonic parser could not extract Gemma-format
        # tool calls, so the agent gave up after 1 react step.
        "tool_call_parser": "gemma4",
        "extra_body": None,
        "sglang_extra_args": ["--trust-remote-code", "--reasoning-parser", "gemma4"],
    },
    "gemma4-26b-a4b-fp8": {
        "model_path": "RedHatAI/gemma-4-26B-A4B-it-FP8-Dynamic",
        "architecture": "moe",  # Mixture-of-Experts with hybrid attention (sliding window + full)
        "params_total": "26B",
        "params_active": "4B",  # 3.8B active per token via MoE routing
        "quantization": "fp8",  # FP8-Dynamic, community quant by RedHatAI
        "tool_call_parser": "gemma4",
        "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
        "sglang_extra_args": [
            "--trust-remote-code",
            "--mem-fraction-static", "0.75",   # cookbook-recommended for MoE path
            "--reasoning-parser", "gemma4",
        ],
    },
    "deepseek-coder-v2-lite": {
        # DeepSeek-Coder-V2-Lite-Instruct: 16B MoE (2.4B active), MIT license, no gating.
        # Non-Qwen family (DeepSeek architecture). SGLang 0.5.10 has native support
        # via --tool-call-parser deepseekv3. Replaces SWE-Dev-9B (which is GLM-based
        # and SGLang 0.5.10 doesn't support ChatGLMForConditionalGeneration).
        # BF16 fit on 48 GB A6000 Ada: ~38 GB total (32 GB weights + 6 GB KV/overhead).
        "model_path": "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct",
        "architecture": "moe",
        "params_total": "16B",
        "params_active": "2.4B",
        "quantization": None,           # BF16
        "tool_call_parser": "deepseekv3",
        "extra_body": None,
        "sglang_extra_args": ["--trust-remote-code"],
    },
    "swe-dev-9b": {
        # SWE-Dev-9B: SWE-bench-tuned via SFT on GLM-4-9B-Chat base (THUDM/SWE-Dev paper 2506.07636).
        # Non-Qwen family (GLM-4 architecture from Zhipu AI). MIT license, no gating.
        # Sizing: 9B params × BF16 = ~18 GB weights + ~6 GB KV cache + ~3 GB SGLang overhead ≈ 27 GB total.
        # Fits comfortably on 48 GB A5000 (atlas24) with headroom for CUDA graphs.
        "model_path": "THUDM/SWE-Dev-9B",
        "architecture": "dense",
        "params_total": "9B",
        "params_active": "9B",
        "quantization": None,  # BF16 (no quantized variant published)
        "tool_call_parser": "glm",  # SGLang 0.5.10 exposes 'glm' (base GLM-4), not 'glm4'
        "extra_body": None,          # GLM-4 has no thinking-mode toggle equivalent to Qwen's
        "sglang_extra_args": ["--trust-remote-code"],
    },
}

# Toy experiment model (for development/energy curves)
TOY_MODEL = {
    "model_path": "Qwen/Qwen3.5-9B",
    "architecture": "dense",
    "params_total": "9B",
    "params_active": "9B",
    "quantization": None,
    "tool_call_parser": "qwen3_coder",
    "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
    "sglang_extra_args": ["--trust-remote-code"],
}

# --- Topologies (Kim et al. 2025 taxonomy) ---
TOPOLOGIES = ["sas", "independent", "centralized", "decentralized", "hybrid"]

# --- Benchmarks (all agentic, spanning parallelizable → sequential → reasoning) ---
BENCHMARKS = ["qampari", "workbench", "browsecomp_plus", "plancraft", "math"]
# qampari: list-answer QA with breadth bottleneck (parallelizable retrieval)
# workbench: 16-tool procedural benchmark (stateful tool composition)
# browsecomp_plus: deep retrieval over 100K corpus
# plancraft: sequential planning (negative control)
# math: Hendrycks MATH Level-5 with stateful Python interpreter
# (reasoning-heavy; fills the Du et al. debate-helps gap left by retrieval-only matrix)

# --- Experiment parameters (Kim et al. 2025 defaults) ---
N_AGENTS = 3           # M=3 (Du et al. standard; Kim et al. primary)
N_REPS = 5             # repetitions per config (Wilcoxon n>=5)
N_WARMUP = 5           # warmup calls before measurement
MAX_TOKENS = 4096      # per-call generation limit (agentic tasks need more)
MAX_REACT_STEPS = 10   # SAS/Independent max iterations (Kim: "max 10 iterations")

# Per-topology structure (Kim et al. 2025, Table 2 / Appendix A)
# With early stopping: rounds are a max cap (agents stop when converged),
# and per-round step budget matches SAS so agents can do as much work as needed.
# Centralized: "3 sub-agents, 1 orchestrator, max 5 rounds"
CENTRALIZED_ROUNDS = 5
CENTRALIZED_WORKER_STEPS = MAX_REACT_STEPS  # let workers use as many steps as needed

# Decentralized: "3 agents, 3 debate rounds"
# Round count is 2 (not 3) because the initial independent phase counts as round 1
DECENTRALIZED_ROUNDS = 2
DECENTRALIZED_DEBATE_STEPS = MAX_REACT_STEPS  # let debaters use as many steps as needed

# Hybrid: centralized structure + limited peer communication
HYBRID_ROUNDS = 5                       # max orchestration rounds (early stop applies)
HYBRID_WORKER_STEPS = MAX_REACT_STEPS   # let workers use as many steps as needed
N_PEER_ROUNDS = 1                       # peer debate rounds per orchestrator round

# --- Temperature ---
# T>0 critical for MAS diversity; T=0 makes agents identical → debate is a no-op
SAS_TEMP = 0.0         # deterministic baseline
INDEPENDENT_TEMP = 0.7 # diversity via sampling (M3MAD precedent)
DEBATE_TEMP = 0.5      # Centralized workers / Decentralized debaters
PEER_TEMP = 0.5        # Hybrid peer debate temperature
ORCHESTRATOR_TEMP = 0.0  # orchestrator decisions are deterministic
BASE_SEED = 42

# --- SGLang server ---
SGLANG_PORT = 30000
SGLANG_URL = f"http://localhost:{SGLANG_PORT}/v1"
SGLANG_API_KEY = "EMPTY"
SGLANG_MEM_FRACTION = 0.80
SGLANG_CONTEXT_LENGTH = int(os.environ.get(
    "SGLANG_CONTEXT_LENGTH", 131072
))  # A6000: 131072, A5000: 49152 (set via env var in sbatch)

# --- Transcript logging ---
# When True, every LLM call's request messages and response content are
# attached to the call_record metadata for downstream inspection. This grows
# result file size ~10x and is intended only for spot-check / debug runs.
# Toggled by --save-transcripts in run_experiments.py (sets MAS_SAVE_TRANSCRIPTS env).
SAVE_TRANSCRIPTS = bool(int(os.environ.get("MAS_SAVE_TRANSCRIPTS", "0")))

# --- Tool-latency injection (deep-research energy side study) ---
# When TOOL_LATENCY_S > 0, each tool execution is padded with sleep so its
# TOTAL latency reaches TOOL_LATENCY_S seconds (emulating a slow web API;
# local BM25 returns in ~0.17s). The padding happens inside the metered
# tool_execution window, so idle-during-wait energy is charged to the tool
# phase exactly as a real remote call would be. TOOL_LATENCY_JITTER is a
# uniform +/- fraction applied per call (e.g. 0.3 -> U[0.7t, 1.3t]).
TOOL_LATENCY_S = float(os.environ.get("MAS_TOOL_LATENCY_S", "0"))
TOOL_LATENCY_JITTER = float(os.environ.get("MAS_TOOL_LATENCY_JITTER", "0"))

# --- Cluster storage root (unused by the runner; kept for reference) ---
CLUSTER_STORAGE = "/atlas2/u"       # /atlas2/u/$USER/
HF_CACHE_DIR = "mas_project/hf_cache"           # relative to user storage
RESULTS_DIR = "mas_project/mas-energy/results"
LOGS_DIR = "mas_project/mas-energy/logs"
