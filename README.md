# mas-energy

Code, configurations, and data for *Depth Over Breadth: The Energy Cost of
Scaling Multi-Agent LLM Systems*.

The experiments sweep depth (K tool-calling steps), breadth (N agents), and
communication rounds (R) for four topologies (SAS, Independent, Centralized,
Decentralized) on four agentic benchmarks (FanOutQA, WorkBench, BrowseComp-Plus,
SWE-bench-Lite). The GPU energy of every LLM call is measured with NVML hardware
counters.

## Layout

| path | contents |
|------|----------|
| `code/run_experiments.py` | entry point; runs one or more cells against a running SGLang server |
| `code/topologies.py` | the four topologies |
| `code/llm.py` | SGLang client and ReAct loop |
| `code/energy.py` | NVML energy counter reads |
| `code/config.py` | models, temperatures, seeds, context rules |
| `code/prompts.py` | role prompts |
| `code/benchmarks*.py` | benchmark adapters and scorers |
| `code/eval_swebench.py` | official SWE-bench harness on saved patches |
| `code/batch_coeff_probe.py`, `code/measure_batch_energy.py` | batching probes (appendix) |
| `scripts/run_cell.sh` | starts a server, runs experiments, stops the server |
| `final_data/` | processed CSVs used for the figures and tables |
| `requirements-cluster-freeze.txt` | pinned packages of the environment the 9B grid ran in |

`final_data/qwen9b_percell/*_{k,n,r}_sweep.csv` hold per-cell means for the
Qwen3.5-9B K/N/R sweeps. `final_data/cross_model/per_record_master.csv` has one row
per task for all three models with per-phase energy and harness accuracy.
`final_data/kxn_grid/` is the joint K x N grid used in the configuration-search
section. `README.md`, `ERA_MAP.md`, and `COVERAGE.md` inside `final_data/` describe
how the CSVs were built and the serving configuration of each cell.

Raw per-task run records (JSONL with transcripts, several hundred GB) are not
included.

## Setup

```
conda create -n mas-energy python=3.11 && conda activate mas-energy
pip install -r requirements-cluster-freeze.txt
```

The environment file pins sglang 0.5.10, torch 2.9.1+cu128, and transformers
5.3.0. The MoE grid used sglang 0.5.15 and the Gemma grid 0.5.14 (needed for the
`gemma4` model class and tool-call parser).

Benchmark data is read locally at run time:

- FanOutQA: `pip install fanoutqa`. The adapter caches Wikipedia pages on disk
  (about 9 GB) on first use.
- WorkBench: clone https://github.com/olly-styles/WorkBench and set
  `WORKBENCH_PATH`.
- BrowseComp-Plus: `Tevatron/browsecomp-plus` and `Tevatron/browsecomp-plus-corpus`
  on HuggingFace (`HF_HOME`). The adapter decrypts the questions and builds a
  BM25 index on first load.
- SWE-bench-Lite: `princeton-nlp/SWE-bench_Lite`. Clone the task repositories
  listed at the top of `benchmarks_swebench.py` and set `SWEBENCH_REPOS`. Scoring
  runs the official harness in Docker through `code/eval_swebench.py`.

SGLang pulls the models from HuggingFace: `Qwen/Qwen3.5-9B` (BF16),
`Qwen/Qwen3.5-35B-A3B-FP8`, `google/gemma-4-31B-it-qat-w4a16`.

## Running experiments

```
export HF_HOME=... WORKBENCH_PATH=... SWEBENCH_REPOS=...
scripts/run_cell.sh toy --benchmarks swebench --topologies decentralized \
    --max-react-steps 10 --rounds 2 --n-agents 3 --n-tasks 50 --n-reps 1 \
    --output-dir results/my_run
```

`toy` is Qwen3.5-9B. `qwen35b-a3b-fp8` and `gemma4-31b-qat-w4a16` are the other
two models (keys in `config.py`). The script reads the model path, tool-call
parser, and extra server flags from `config.py`, applies the memory fraction,
context length, and quantization used for the paper, waits for the server to
become healthy, runs `run_experiments.py`, and shuts the server down. Override
with `MEM_FRACTION`, `CONTEXT_LENGTH`, `SGLANG_PORT`, `SGLANG_EXTRA`. Put
`--dry-run` after the model key to print the server and runner commands without
running them. The GPU must be allocated exclusively because energy is read from
the device counter.

`--benchmarks`, `--topologies`, and `--max-react-steps` accept lists, so one call
can run a whole sweep against one server:

```
scripts/run_cell.sh toy --benchmarks fanoutqa workbench --topologies sas independent \
    --max-react-steps 1 2 3 5 7 10 15 20 30 50 --n-tasks 50 --output-dir results/ksweep
```

On a cluster, wrap that line in your scheduler's job script with one GPU per job
and a distinct `SGLANG_PORT` per job.

Before measuring, `run_experiments.py` makes 5 warmup calls and records idle GPU
power for 10 s. Tasks run one at a time. Each cell produces one JSONL with
per-task energy, tokens, calls, accuracy, and transcript, plus a metadata file.
Runs can be resumed: pointing a job at an existing output directory only runs the
missing tasks.

Grid used in the paper: K in {1, 2, 3, 5, 7, 10, 15, 20, 30, 50} (plus 100 and 200
on SWE-bench) at N = 3, R = 2; N in {1, 2, 3, 4, 5, 10} at K = 20 (Independent) or
K = 10, R = 2; R in {1, 2, 3, 5, 6, 10, 15, 30} with K = 30 / R; 50 tasks per cell
for K, 100 for N and R. Temperatures, seeds, the context truncation rule, and
per-model launch flags are given in the paper appendix and in `config.py` and
`llm.py`.
