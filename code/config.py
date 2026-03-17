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

# --- Benchmarks (all agentic, spanning parallelizable → sequential) ---
BENCHMARKS = ["workbench", "browsecomp_plus", "plancraft"]
# LocalSearchBench replacement TBD — needs offline parallelizable benchmark

# --- Experiment parameters ---
N_AGENTS = 3           # M=3 (Du et al. standard; Kim et al. primary)
N_ROUNDS = 2           # coordination/debate rounds
N_PEER_ROUNDS = 1      # peer debate rounds within each orchestrator round (Hybrid only)
MAX_PEER_STEPS = 10    # max ReAct steps per peer debate round
N_REPS = 5             # repetitions per config (Wilcoxon n>=5)
N_WARMUP = 5           # warmup calls before measurement
MAX_TOKENS = 4096      # per-call generation limit (agentic tasks need more)
MAX_REACT_STEPS = 20   # max tool-calling steps per ReAct loop

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
SGLANG_CONTEXT_LENGTH = 32768

# --- Cluster paths (Stanford SC / atlas) ---
CLUSTER_STORAGE = "/atlas2/u"       # /atlas2/u/$USER/
HF_CACHE_DIR = "mas_project/hf_cache"           # relative to user storage
RESULTS_DIR = "mas_project/mas-energy/results"
LOGS_DIR = "mas_project/mas-energy/logs"
