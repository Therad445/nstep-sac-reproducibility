from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def env_name_from_stem(stem: str) -> str:
    if "_sac_" in stem:
        return stem.split("_sac_", maxsplit=1)[0]
    return "all"


def label_from_stem(stem: str) -> str:
    if "_n" in stem:
        n_part = stem.split("_n", maxsplit=1)[1].split("_", maxsplit=1)[0]
        return f"SAC n={n_part}"
    return stem


def pretty_env_name(env_name: str) -> str:
    mapping = {
        "pendulum": "Pendulum-v1",
        "halfcheetah": "HalfCheetah-v5",
    }
    return mapping.get(env_name, env_name)


def plot_group(paths: list[Path], out_dir: Path, env_name: str) -> None:
    plt.figure(figsize=(10, 6))

    for path in sorted(paths):
        df = pd.read_csv(path)
        if "step" not in df.columns or "eval_return" not in df.columns:
            continue

        plt.plot(
            df["step"],
            df["eval_return"],
            marker="o",
            linewidth=2,
            label=label_from_stem(path.stem),
        )

    title = f"{pretty_env_name(env_name)} single-seed n-step SAC ablation"
    plt.title(title)
    plt.xlabel("Environment steps")
    plt.ylabel("Mean deterministic evaluation return")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    out_path = out_dir / f"{env_name}_learning_curves.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved plot to {out_path}")


def main() -> None:
    args = parse_args()
    input_dir = args.input
    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = sorted(input_dir.glob("*_sac_n*_seed*.csv"))
    if not paths:
        raise FileNotFoundError(f"No experiment CSV files found in {input_dir}")

    groups: dict[str, list[Path]] = {}
    for path in paths:
        env_name = env_name_from_stem(path.stem)
        groups.setdefault(env_name, []).append(path)

    for env_name, group_paths in sorted(groups.items()):
        plot_group(group_paths, out_dir, env_name)


if __name__ == "__main__":
    main()
