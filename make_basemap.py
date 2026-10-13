#!/usr/bin/env python
"""Rasterize the Natural Earth land polygons into the reference base map.

Land and sea pixels get the colors from the [colors] table of config.toml,
the same ones the LLM script uses. A pixel is land when its
center falls inside a land polygon, which matches the question the LLM script
asks about the same pixel. Pass --all-touched to also mark pixels that a
polygon merely touches; this keeps more small islands.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import shapefile
from rasterio import features
from rasterio.transform import from_bounds

import mapgrid

DEFAULT_SHAPEFILE = Path("data/ne_110m_land/ne_110m_land.shp")
DEFAULT_OUT = Path("maps/basemap.png")
DEFAULT_CONFIG = Path("config.toml")


def rasterize_land(shp_path: Path, width: int, height: int, all_touched: bool) -> np.ndarray:
    """Return a boolean (height, width) array, True where the pixel is land."""
    if not shp_path.is_file():
        raise SystemExit(
            f"shapefile not found: {shp_path}\n"
            "Download it first: ./download_data.sh (see README.md)."
        )
    reader = shapefile.Reader(str(shp_path))
    shapes = [(shape.__geo_interface__, 1) for shape in reader.shapes()]
    transform = from_bounds(-180.0, -90.0, 180.0, 90.0, width, height)
    mask = features.rasterize(
        shapes,
        out_shape=(height, width),
        transform=transform,
        fill=0,
        all_touched=all_touched,
        dtype="uint8",
    )
    return mask.astype(bool)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--shapefile", type=Path, default=DEFAULT_SHAPEFILE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="config file that holds the [colors] table")
    parser.add_argument("--width", type=int, default=mapgrid.WIDTH)
    parser.add_argument("--height", type=int, default=mapgrid.HEIGHT)
    parser.add_argument("--all-touched", action="store_true", help="mark a pixel as land if any polygon touches it")
    args = parser.parse_args()

    try:
        colors = mapgrid.load_colors(args.config)
    except (OSError, ValueError) as exc:
        raise SystemExit(f"{args.config}: {exc}")

    land = rasterize_land(args.shapefile, args.width, args.height, args.all_touched)
    img = np.empty((args.height, args.width, 3), dtype=np.uint8)
    img[:, :] = colors.sea
    img[land] = colors.land
    mapgrid.save_map_atomic(img, args.out)

    frac = land.mean() * 100.0
    print(f"wrote {args.out} ({args.width}x{args.height}), land pixels: {land.sum()} ({frac:.1f}%)")


if __name__ == "__main__":
    main()
