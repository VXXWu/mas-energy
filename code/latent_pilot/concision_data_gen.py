"""Generate (verbose, concise) SFT pairs for concision-tuning an agent.

Strategy: for each QAMPARI training question, run the text pipeline to
collect (context, verbose_answer) pairs, then programmatically produce a
concise entity-list version (regex-extract entities from the verbose
answer, join with commas). Train the model to emit the concise version
in one SFT step — no latent mechanism, just shorter output conditioning.

Complementary to grammar constraints: grammar forces short output at
decode time; concision tuning biases the model to produce short output
without needing stop-strings.

Usage:
    python concision_data_gen.py --n-tasks 500 --out-path concise_pairs.jsonl
"""
import argparse
import json
import os
import re
import sys

CODE_DIR = os.environ.get(
    "MAS_ENERGY_CODE",
    f"/atlas2/u/{os.environ.get('USER', 'vincewu8')}/mas_project/mas-energy/code",
)
sys.path.insert(0, CODE_DIR)


def extract_entity_list(text):
    """Pull clean comma-separated entity list from verbose text.

    Strategy: split on commas and newlines, drop prose items (start with
    articles, too long, or have more than 3 words), lowercase-strip,
    re-join the surviving items. Takes the first such list-like chunk
    found.
    """
    # Remove markdown bold/italic/code
    t = re.sub(r'[*_#`]', '', text)
    # Split on commas or newlines
    items = re.split(r'[,\n]', t)
    cleaned = []
    for item in items:
        item = item.strip().strip('.!?:;()[]{}').strip()
        if 1 <= len(item) <= 80 and item.count(" ") <= 4:
            if not item.lower().startswith(('the ', 'a ', 'an ', 'and ', 'based ', 'here ', 'these ')):
                cleaned.append(item)
    # Dedup preserving order
    seen = set()
    uniq = []
    for x in cleaned:
        key = x.lower()
        if key not in seen:
            seen.add(key)
            uniq.append(x)
    return ", ".join(uniq) if uniq else text.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-tasks", type=int, default=500)
    ap.add_argument("--split", type=str, default="train")
    ap.add_argument("--out-path", type=str, required=True)
    ap.add_argument("--model-name", type=str, default="Qwen/Qwen3-8B")
    ap.add_argument("--max-react-steps", type=int, default=10)
    ap.add_argument("--data-dir", type=str, required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from benchmarks_qampari import QampariBenchmark
    from latent_pilot.agentic_latentmas import build_sas_prompt, text_react_loop

    print(f"Loading {args.model_name}...")
    tok = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.bfloat16).to("cuda")
    model.eval()

    class MW:
        tokenizer = tok
        model = model
    mw = MW()

    bench = QampariBenchmark(data_dir=args.data_dir)
    tasks = bench.load_tasks(n_tasks=args.n_tasks, seed=args.seed, split=args.split)
    print(f"Loaded {len(tasks)} qampari {args.split} tasks")

    out_path = os.path.abspath(args.out_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if os.path.exists(out_path):
        os.remove(out_path)

    kept = 0
    for i, task in enumerate(tasks):
        q = task["question_text"]
        executor, cleanup = bench.make_executor(task)
        try:
            msgs = build_sas_prompt(q)
            verbose = text_react_loop(mw, msgs, executor,
                                       max_steps=args.max_react_steps, temperature=0.0)
            cleanup()
            concise = extract_entity_list(verbose)
            # Only keep pairs where concise is meaningfully shorter than verbose
            if len(concise) < len(verbose) * 0.6 and len(concise) > 20:
                rec = {
                    "task_id": task.get("qid", task.get("id", i)),
                    "question": q,
                    "messages_pre_answer": msgs,
                    "verbose_answer": verbose,
                    "concise_target": concise,
                    "compression": 1 - len(concise) / len(verbose),
                }
                with open(out_path, "a") as f:
                    f.write(json.dumps(rec) + "\n")
                kept += 1
        except Exception as e:
            cleanup()
            print(f"[{i}] error: {e}")
            continue
        if (i + 1) % 20 == 0:
            print(f"[{i+1}/{len(tasks)}] kept={kept}")

    print(f"Done. Saved {kept}/{len(tasks)} pairs to {out_path}")


if __name__ == "__main__":
    main()
