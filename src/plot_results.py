from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="results/raw")
    parser.add_argument("--out", default="results/plots/learning_curves.png")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    files = sorted(input_dir.glob("*.csv"))
    if not files:
        raise SystemExit(f"No csv logs found in {input_dir}")

    plt.figure(figsize=(9, 5))
    for path in files:
        df = pd.read_csv(path)
        if "step" not in df or "eval_return" not in df:
            continue
        plt.plot(df["step"], df["eval_return"], label=path.stem)

    plt.xlabel("Environment steps")
    plt.ylabel("Evaluation return")
    plt.title("SAC n-step ablation")
    plt.legend()
    plt.tight_layout()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=160)
    print(f"Saved plot to {out}")


if __name__ == "__main__":
    main()
