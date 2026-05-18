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

# --- Cluster paths (Stanford SC / atlas) ---
CLUSTER_STORAGE = "/atlas2/u"       # /atlas2/u/$USER/
HF_CACHE_DIR = "mas_project/hf_cache"           # relative to user storage
RESULTS_DIR = "mas_project/mas-energy/results"
LOGS_DIR = "mas_project/mas-energy/logs"
