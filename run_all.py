"""
Run the complete pipeline end to end.

    python run_all.py              # everything (several hours on a laptop)
    python run_all.py --skip-fetch # data already downloaded
    python run_all.py --only train train_classifiers

Steps (in order): fetch -> prep -> test -> eda -> imputation -> train ->
train_classifiers -> errors -> explain -> visuals -> robustness -> ablation -> plots
"""
import argparse
import subprocess
import sys
import time

STEPS = [
    ("fetch", ["src/fetch_external.py"]),
    ("prep", ["src/data_prep.py"]),
    ("test", ["tests/test_leakage.py"]),
    ("eda", ["src/eda.py"]),
    ("imputation", ["src/imputation_study.py"]),
    ("train", ["src/train.py"]),
    ("train_classifiers", ["src/train_classifiers.py"]),
    ("errors", ["src/analysis_errors.py"]),
    ("explain", ["src/analysis_explain.py"]),
    ("visuals", ["src/analysis_visuals.py"]),
    ("robustness", ["src/analysis_robustness.py"]),
    ("ablation", ["src/analysis_ablation.py"]),
    ("plots", ["src/analysis_plots.py"]),
]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-fetch", action="store_true")
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    for name, cmd in STEPS:
        if a.only and name not in a.only:
            continue
        if a.skip_fetch and name == "fetch":
            continue
        t = time.time()
        print(f"\n>>> {name}: {' '.join(cmd)}", flush=True)
        r = subprocess.run([sys.executable, "-W", "ignore"] + cmd)
        if r.returncode:
            sys.exit(f"step {name} failed")
        print(f"<<< {name} done in {(time.time() - t) / 60:.1f} min", flush=True)
