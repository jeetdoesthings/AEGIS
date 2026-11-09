import subprocess, sys, os, json, glob
import numpy as np
import pandas as pd

def run_cmd(cmd):
    print(f"\n>>> Running: {cmd}", flush=True)
    res = subprocess.run(cmd, shell=True)
    if res.returncode != 0:
        print(f"Warning: Command failed with code {res.returncode}", flush=True)

PYTHON = ".venv/bin/python"

configs_to_run = [
    # 1. finetune
    ("finetune", f"{PYTHON} cl2.py --methods finetune --seeds 3 --seed_start 0 --eval_split val --out phase1/tune_finetune"),
    
    # 2. replay (buf 50, 200, 1000)
    ("replay_buf50", f"{PYTHON} cl2.py --methods replay --buf 50 --seeds 3 --seed_start 0 --eval_split val --out phase1/tune_replay_buf50"),
    ("replay_buf200", f"{PYTHON} cl2.py --methods replay --buf 200 --seeds 3 --seed_start 0 --eval_split val --out phase1/tune_replay_buf200"),
    ("replay_buf1000", f"{PYTHON} cl2.py --methods replay --buf 1000 --seeds 3 --seed_start 0 --eval_split val --out phase1/tune_replay_buf1000"),
    
    # 3. lora (r 4, 8, 16)
    ("lora_r4", f"{PYTHON} cl2.py --methods lora --r 4 --seeds 3 --seed_start 0 --eval_split val --out phase1/tune_lora_r4"),
    ("lora_r8", f"{PYTHON} cl2.py --methods lora --r 8 --seeds 3 --seed_start 0 --eval_split val --out phase1/tune_lora_r8"),
    ("lora_r16", f"{PYTHON} cl2.py --methods lora --r 16 --seeds 3 --seed_start 0 --eval_split val --out phase1/tune_lora_r16"),
    
    # 4. gate:conflict (beta in 0.25, 0.5, 1.0; decay in 1.0, 0.9)
    ("gate_conf_b025_d10", f"{PYTHON} cl2.py --methods gate:conflict:0.25 --decay 1.0 --seeds 3 --seed_start 0 --eval_split val --out phase1/tune_gate_conf_b025_d10"),
    ("gate_conf_b025_d09", f"{PYTHON} cl2.py --methods gate:conflict:0.25 --decay 0.9 --seeds 3 --seed_start 0 --eval_split val --out phase1/tune_gate_conf_b025_d09"),
    ("gate_conf_b05_d10", f"{PYTHON} cl2.py --methods gate:conflict:0.5 --decay 1.0 --seeds 3 --seed_start 0 --eval_split val --out phase1/tune_gate_conf_b05_d10"),
    ("gate_conf_b05_d09", f"{PYTHON} cl2.py --methods gate:conflict:0.5 --decay 0.9 --seeds 3 --seed_start 0 --eval_split val --out phase1/tune_gate_conf_b05_d09"),
    ("gate_conf_b10_d10", f"{PYTHON} cl2.py --methods gate:conflict:1.0 --decay 1.0 --seeds 3 --seed_start 0 --eval_split val --out phase1/tune_gate_conf_b10_d10"),
    ("gate_conf_b10_d09", f"{PYTHON} cl2.py --methods gate:conflict:1.0 --decay 0.9 --seeds 3 --seed_start 0 --eval_split val --out phase1/tune_gate_conf_b10_d09"),
    
    # 5. gate:magnitude (beta in 0.25, 0.5, 1.0; decay in 1.0, 0.9)
    ("gate_mag_b025_d10", f"{PYTHON} cl2.py --methods gate:magnitude:0.25 --decay 1.0 --seeds 3 --seed_start 0 --eval_split val --out phase1/tune_gate_mag_b025_d10"),
    ("gate_mag_b025_d09", f"{PYTHON} cl2.py --methods gate:magnitude:0.25 --decay 0.9 --seeds 3 --seed_start 0 --eval_split val --out phase1/tune_gate_mag_b025_d09"),
    ("gate_mag_b05_d10", f"{PYTHON} cl2.py --methods gate:magnitude:0.5 --decay 1.0 --seeds 3 --seed_start 0 --eval_split val --out phase1/tune_gate_mag_b05_d10"),
    ("gate_mag_b05_d09", f"{PYTHON} cl2.py --methods gate:magnitude:0.5 --decay 0.9 --seeds 3 --seed_start 0 --eval_split val --out phase1/tune_gate_mag_b05_d09"),
    ("gate_mag_b10_d10", f"{PYTHON} cl2.py --methods gate:magnitude:1.0 --decay 1.0 --seeds 3 --seed_start 0 --eval_split val --out phase1/tune_gate_mag_b10_d10"),
    ("gate_mag_b10_d09", f"{PYTHON} cl2.py --methods gate:magnitude:1.0 --decay 0.9 --seeds 3 --seed_start 0 --eval_split val --out phase1/tune_gate_mag_b10_d09"),
    
    # 6. gate:layer_cos (beta in 0.5, 1.0; decay=1.0)
    ("gate_cos_b05_d10", f"{PYTHON} cl2.py --methods gate:layer_cos:0.5 --decay 1.0 --seeds 3 --seed_start 0 --eval_split val --out phase1/tune_gate_cos_b05_d10"),
    ("gate_cos_b10_d10", f"{PYTHON} cl2.py --methods gate:layer_cos:1.0 --decay 1.0 --seeds 3 --seed_start 0 --eval_split val --out phase1/tune_gate_cos_b10_d10"),
]

for name, cmd in configs_to_run:
    # Check if results already exist
    res_path = f"outputs/phase1/results_tune_{name}.json" if "tune_" in cmd else f"outputs/phase1/results_{name}.json"
    run_cmd(cmd)

print("\nAll tuning runs executed. Aggregating results into outputs/phase1/tuning_val.csv...")

# Now gather all results from outputs/phase1/results_ewc_val_lam*.json and results_tune_*.json
all_tuning_files = sorted(glob.glob("outputs/phase1/results_ewc_val_lam*.json") + glob.glob("outputs/phase1/results_tune_*.json"))

records = []
for f in all_tuning_files:
    tag = os.path.basename(f).replace("results_", "").replace(".json", "")
    cfg_file = f.replace("results_", "config_")
    cfg = json.load(open(cfg_file)) if os.path.exists(cfg_file) else {}
    data = json.load(open(f))
    
    for m, rs in data.items():
        if any(r.get("diverged", False) for r in rs):
            records.append({
                "tag": tag, "method": m, "config": cfg,
                "ACC_mean": None, "ACC_std": None,
                "BWT_mean": None, "BWT_std": None,
                "FWT_mean": None, "FWT_std": None,
                "NEW_mean": None, "NEW_std": None,
                "diverged": True
            })
        else:
            accs = [r["ACC"] * 100 for r in rs]
            bwts = [r["BWT"] * 100 for r in rs]
            fwts = [r["FWT"] * 100 for r in rs]
            news = [r["NEW"] * 100 for r in rs]
            records.append({
                "tag": tag, "method": m, "config": cfg,
                "ACC_mean": float(np.mean(accs)), "ACC_std": float(np.std(accs)),
                "BWT_mean": float(np.mean(bwts)), "BWT_std": float(np.std(bwts)),
                "FWT_mean": float(np.mean(fwts)), "FWT_std": float(np.std(fwts)),
                "NEW_mean": float(np.mean(news)), "NEW_std": float(np.std(news)),
                "diverged": False
            })

# Save CSV
csv_rows = []
for r in records:
    c = r["config"]
    if r["diverged"]:
        csv_rows.append({
            "tag": r["tag"], "method": r["method"],
            "params": f"diverged",
            "ACC": "DIVERGED", "BWT": "--", "FWT": "--", "NEW": "--"
        })
    else:
        # summarize relevant params
        p_str = ""
        if "ewc" in r["method"]: p_str = f"lambda={c.get('ewc_lambda')}"
        elif "replay" in r["method"]: p_str = f"buf={c.get('buf')}"
        elif "lora" in r["method"]: p_str = f"r={c.get('r')}"
        elif "gate" in r["method"]: p_str = f"beta={c.get('beta')},decay={c.get('decay')}"
        csv_rows.append({
            "tag": r["tag"], "method": r["method"], "params": p_str,
            "ACC": f"{r['ACC_mean']:.2f} +- {r['ACC_std']:.2f}",
            "BWT": f"{r['BWT_mean']:.2f} +- {r['BWT_std']:.2f}",
            "FWT": f"{r['FWT_mean']:.2f} +- {r['FWT_std']:.2f}",
            "NEW": f"{r['NEW_mean']:.2f} +- {r['NEW_std']:.2f}",
        })

df_csv = pd.DataFrame(csv_rows)
csv_out = "outputs/phase1/tuning_val.csv"
df_csv.to_csv(csv_out, index=False)
print(f"Saved {csv_out}")
print("\n" + df_csv.to_string())

# Determine locked configuration for each method family
# Families: finetune, ewc, replay, lora, gate:conflict, gate:magnitude, gate:layer_cos
families = ["finetune", "ewc", "replay", "lora", "gate:conflict", "gate:magnitude", "gate:layer_cos"]
locked = {}

for fam in families:
    cand = [r for r in records if not r["diverged"] and r["method"].startswith(fam)]
    if not cand:
        print(f"Warning: No valid candidate for {fam}")
        continue
    # sort by ACC_mean desc, then BWT_mean desc, then NEW_mean desc
    cand.sort(key=lambda x: (x["ACC_mean"], x["BWT_mean"], x["NEW_mean"]), reverse=True)
    best = cand[0]
    
    # Check if any other candidate is a tie (within 1 std)
    # Rule: Two methods are tied if their mean ± std intervals overlap.
    # Break ties with BWT, then NEW
    tied = []
    best_low = best["ACC_mean"] - best["ACC_std"]
    best_high = best["ACC_mean"] + best["ACC_std"]
    for c in cand:
        c_low = c["ACC_mean"] - c["ACC_std"]
        c_high = c["ACC_mean"] + c["ACC_std"]
        if max(best_low, c_low) <= min(best_high, c_high): # overlap
            tied.append(c)
    
    # Pick best among tied by BWT_mean desc, then NEW_mean desc
    tied.sort(key=lambda x: (x["BWT_mean"], x["NEW_mean"]), reverse=True)
    chosen = tied[0]
    
    c_info = chosen["config"]
    locked[fam] = {
        "method": chosen["method"],
        "tag": chosen["tag"],
        "ACC": f"{chosen['ACC_mean']:.2f} +- {chosen['ACC_std']:.2f}",
        "BWT": f"{chosen['BWT_mean']:.2f} +- {chosen['BWT_std']:.2f}",
        "NEW": f"{chosen['NEW_mean']:.2f} +- {chosen['NEW_std']:.2f}",
        "params": {
            "ewc_lambda": c_info.get("ewc_lambda"),
            "buf": c_info.get("buf"),
            "r": c_info.get("r", 8),
            "decay": c_info.get("decay", 1.0),
        }
    }

locked_path = "configs/locked_perm5.json"
os.makedirs("configs", exist_ok=True)
with open(locked_path, "w") as f:
    json.dump(locked, f, indent=2)

print(f"\nSaved locked configurations to {locked_path}:")
print(json.dumps(locked, indent=2))
