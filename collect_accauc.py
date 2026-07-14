"""Collect acc-AUC (and CPR-AUC) for every node method across the 12 cells.

Reads the patched-eval pkls under results/<dir>_accauc/ (each has both `acc_auc`
and `area_under`). Prints one table per metric: rows = task/model cell, cols = method.
Run:  .venv/bin/python collect_accauc.py
"""
import pickle, os, sys

CELLS = [
    ("gpt2", "ioi"), ("qwen2.5", "ioi"), ("qwen2.5", "mcqa"),
    ("gemma2", "ioi"), ("gemma2", "mcqa"), ("gemma2", "arc_easy"),
    ("llama3", "ioi"), ("llama3", "mcqa"), ("llama3", "arithmetic_addition"),
    ("llama3", "arithmetic_subtraction"), ("llama3", "arc_easy"), ("llama3", "arc_challenge"),
]
# tag -> (output_dir, method_name_saveable)
METHODS = {
    "NAP-IG":    ("napig_ref_accauc",   "EAP-IG-inputs_patching_node"),
    "NAP-local": ("napig_local_accauc", "EAP-IG-inputs-local_patching_node"),
    "IxG(1)":    ("ig1_accauc",         "EAP-IG-inputs_patching_node"),
    "RelP":      ("relp_accauc",        "RelP_patching_node"),
    "RelP-qk":   ("relp_qkgrad_accauc", "RelP-qkgrad_patching_node"),
    "GIM":       ("gim_accauc",         "GIM_patching_node"),
    "AttnRLP":   ("attnrlp_accauc",     "AttnRLP_patching_node"),
}
BASE = "results"


def load(odir, mdir, task, model):
    stask = task.replace("_", "-")
    p = f"{BASE}/{odir}/{mdir}/{stask}_{model}_validation_abs-False.pkl"
    if not os.path.exists(p):
        return None
    try:
        return pickle.load(open(p, "rb"))
    except Exception:
        return None


def table(metric, label):
    hdr = f"{label + ' | cell':28}" + "".join(f"{m:>11}" for m in METHODS)
    print(hdr); print("-" * len(hdr))
    rows = []
    for model, task in CELLS:
        cell = f"{task}/{model}"
        vals = {}
        for m, (od, md) in METHODS.items():
            d = load(od, md, task, model)
            vals[m] = None if d is None else d.get(metric)
        rows.append(vals)
        s = f"{cell:28}" + "".join(
            (f"{vals[m]:>11.4f}" if vals[m] is not None else f"{'--':>11}") for m in METHODS)
        print(s)
    print("-" * len(hdr))
    for m in METHODS:
        done = [v[m] for v in rows if v[m] is not None]
        mean = sum(done) / len(done) if done else float("nan")
        print(f"  {m:12} mean over {len(done):2}/12 = {mean:.4f}")
    print()


if __name__ == "__main__":
    table("acc_auc", "acc-AUC")
    print("=" * 60)
    table("area_under", "CPR-AUC")
