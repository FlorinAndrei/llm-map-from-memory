#!/usr/bin/env python
"""Draw the base map and the LLM maps side by side in one image.

    .venv/bin/python make_composite.py [--config FILE] [--out composite.png] [MAP ...]

Without MAP arguments, every maps/llm_*.png is used. Each tile gets the
model name as a subtitle. The pixels are drawn as they are, with no
smoothing.
"""

from __future__ import annotations

import argparse
import math
import tomllib
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import mapgrid


def model_name(path: Path) -> str:
    stem = path.stem
    return stem[4:] if stem.startswith("llm_") else stem


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=Path("config.toml"))
    parser.add_argument("--out", type=Path, default=Path("composite.png"))
    parser.add_argument("--columns", type=int, default=2)
    parser.add_argument("maps", nargs="*", type=Path)
    args = parser.parse_args()

    with open(args.config, "rb") as fh:
        cfg = tomllib.load(fh)
    base_path = Path(cfg.get("basemap", "maps/basemap.png"))
    llm_paths = args.maps or sorted(Path(cfg.get("out_dir", "maps")).glob("llm_*.png"))

    base = mapgrid.load_map(base_path)
    tiles = [(base, "Natural Earth")]
    for path in llm_paths:
        tiles.append((mapgrid.load_map(path), model_name(path)))

    cols = max(1, min(args.columns, len(tiles)))
    rows = math.ceil(len(tiles) / cols)
    h, w = base.shape[:2]
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 3.4), dpi=150)
    axes = np.atleast_1d(axes).ravel()
    for ax, (img, label) in zip(axes, tiles):
        ax.imshow(img, interpolation="nearest", extent=(-180, 180, -90, 90))
        ax.set_title(label, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)
    for ax in axes[len(tiles):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(args.out)
    print(f"wrote {args.out} with {len(tiles)} tiles")


if __name__ == "__main__":
    main()
