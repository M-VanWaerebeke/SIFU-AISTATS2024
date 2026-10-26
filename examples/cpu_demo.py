"""Train and forget one client with the real SIFU implementation, offline."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def run_demo(output_dir):
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    common = [sys.executable, str(ROOT / "FU.py"), "--dataset_name", "synthetic",
              "--device", "cpu", "--T", "4", "--M", "4", "--n_sampled", "4",
              "--B", "20", "--n_SGD", "1", "--lr_l", "0.1", "--clip", "10",
              "--epsilon", "1", "--sigma", "0.05", "--seed", "0"]
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    for scheme, policy in [("train", "P0"), ("SIFU", "P12")]:
        subprocess.run(common + ["--unlearn_scheme", scheme, "--forgetting", policy],
                       cwd=output_dir, env=env, check=True)
    checkpoints = list((output_dir / "saved_exp_info/final_model").glob("*SIFU*.pth"))
    if len(checkpoints) != 1:
        raise RuntimeError("Expected one SIFU output checkpoint")
    summary = {"dataset": "synthetic", "clients_before": 4, "forgotten_clients": [0],
               "clients_after": 3, "rounds_per_stage": 4,
               "checkpoint": str(checkpoints[0].relative_to(output_dir)),
               "purpose": "execution smoke test; not paper reproduction"}
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="demo-output")
    run_demo(parser.parse_args().output_dir)
