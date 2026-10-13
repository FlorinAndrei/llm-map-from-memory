"""Shared geometry, colors and file helpers for the world map project.

The map is an equirectangular (plate carree) image. Column 0 is the west edge
(-180 deg), row 0 is the north edge (+90 deg). Each pixel has the same angular
size in both axes when the image is twice as wide as it is tall.
"""

from __future__ import annotations

import os
import re
import tempfile
import tomllib
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
from PIL import Image

WIDTH = 320
HEIGHT = 160

RGB = tuple[int, int, int]

# Smallest allowed luminance gap between any two of the four colors.
MIN_GRAY_GAP = 32


def grayscale(rgb: RGB) -> int:
    """Luminance 0..255 by ITU-R BT.601, as Pillow computes it for mode L."""
    r, g, b = rgb
    return (r * 19595 + g * 38470 + b * 7471 + 0x8000) >> 16


class Colors(NamedTuple):
    """The four pixel colors, read from the [colors] table of config.toml.

    The LLM map uses these exact values as its state, so they must stay
    distinct and must not change while a map is in progress.
    """

    land: RGB
    sea: RGB
    blank: RGB     # not answered yet
    unknown: RGB   # the model gave no usable answer


def colors_from_config(cfg: dict[str, Any]) -> Colors:
    """Build Colors from a parsed config. Raises ValueError on a bad table."""
    table = cfg.get("colors")
    if not isinstance(table, dict):
        raise ValueError("config: a [colors] table with land, sea, blank and unknown is required")
    values: dict[str, RGB] = {}
    for name in Colors._fields:
        rgb = table.get(name)
        ok = (
            isinstance(rgb, list)
            and len(rgb) == 3
            and all(isinstance(c, int) and not isinstance(c, bool) and 0 <= c <= 255 for c in rgb)
        )
        if not ok:
            raise ValueError(f"config: colors.{name} must be a list of three integers from 0 to 255")
        values[name] = (rgb[0], rgb[1], rgb[2])
    if len(set(values.values())) != len(Colors._fields):
        raise ValueError("config: the four colors must all be different")
    by_gray = sorted(values.items(), key=lambda kv: grayscale(kv[1]))
    for (name_a, rgb_a), (name_b, rgb_b) in zip(by_gray, by_gray[1:]):
        gap = grayscale(rgb_b) - grayscale(rgb_a)
        if gap < MIN_GRAY_GAP:
            raise ValueError(
                f"config: colors.{name_a} (gray {grayscale(rgb_a)}) and colors.{name_b} "
                f"(gray {grayscale(rgb_b)}) are too close in grayscale, need a gap of at least {MIN_GRAY_GAP}"
            )
    return Colors(**values)


def load_colors(config_path: str | Path) -> Colors:
    """Read only the [colors] table from a config file."""
    with open(config_path, "rb") as fh:
        return colors_from_config(tomllib.load(fh))


def pixel_to_lonlat(x: int, y: int, width: int = WIDTH, height: int = HEIGHT) -> tuple[float, float]:
    """Return (lon, lat) of the center of pixel (x, y)."""
    lon = -180.0 + (x + 0.5) * 360.0 / width
    lat = 90.0 - (y + 0.5) * 180.0 / height
    return lon, lat


def new_blank(blank: RGB, width: int = WIDTH, height: int = HEIGHT) -> np.ndarray:
    """Return an RGB image array of shape (height, width, 3) filled with the blank color."""
    img = np.empty((height, width, 3), dtype=np.uint8)
    img[:, :] = blank
    return img


def load_map(path: str | Path) -> np.ndarray:
    """Load a PNG as an RGB uint8 array of shape (height, width, 3)."""
    with Image.open(path) as im:
        return np.array(im.convert("RGB"), dtype=np.uint8)


def save_map_atomic(img: np.ndarray, path: str | Path) -> None:
    """Write the image so that the file on disk is always complete.

    The PNG goes to a temporary file in the same directory, then os.replace
    swaps it in. A crash or CTRL-C during the write leaves the old file intact.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as fh:
            Image.fromarray(img, mode="RGB").save(fh, format="PNG", optimize=True)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def mask_of(img: np.ndarray, color: RGB) -> np.ndarray:
    """Boolean mask of the pixels that have exactly this color."""
    return np.all(img == np.array(color, dtype=np.uint8), axis=-1)


def sanitize_name(name: str) -> str:
    """Make a model name safe for use inside a file name."""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", name)
    return cleaned.strip("._-") or "model"
