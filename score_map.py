#!/usr/bin/env python
"""Compare one or more LLM maps with a reference map and print metrics.

    .venv/bin/python score_map.py [--config FILE] [--reference PATH] MAP [MAP ...]

The reference is the base map from config.toml unless --reference is given.
Any map can be the reference, so two LLM maps can be compared with each other.

Only asked pixels count: blank pixels of the map under test are left out, so
a partial map gets a fair score. Unknown pixels (no usable answer) count as
wrong for both classes.

Metrics, and what each one tells about the knowledge of a model:

- Confusion matrix of land and sea, plus land share. The land share next to
  the true land share shows the bias of the model towards one answer.
- IoU (intersection over union) per class and their mean. IoU is the usual
  score for a segmentation; it is low when a class is either missed or
  over-called, so it does not reward a model that answers "land" everywhere.
- Accuracy, balanced accuracy, precision, recall and F1 per class, Cohen's
  kappa and the Matthews correlation coefficient. Kappa and MCC are zero
  for a model that guesses at the base rate and one for a perfect map, so
  they separate knowledge from the luck of a skewed base rate (67% sea).
- Each of the above is also given area-weighted. A pixel row is weighted
  by the cosine of its latitude, so that the Arctic and Antarctic rows do
  not count as much as the equator. The unweighted numbers describe the
  picture; the weighted numbers describe the planet.
- Accuracy by distance from the true coast. A model whose errors sit on the
  coast knows the shapes but not the exact edge (fuzziness). A model with
  errors deep inland or far out at sea does not know the shapes (ignorance).
  The distance is in pixels of the reference map, as rings of neighbors
  (Chebyshev distance): 1 means the pixel touches the other class.
- Coastline length and isolated-pixel share, for the map and the reference.
  A noisy map has a far longer land/sea boundary and many pixels that
  disagree with most of their neighbors.
"""

from __future__ import annotations

import argparse
import math
import sys
import tomllib
from pathlib import Path

import numpy as np

import mapgrid

DISTANCE_BINS = [(1, 1), (2, 2), (3, 4), (5, 8), (9, 16), (17, 10_000)]


# ---------------------------------------------------------------- helpers ----

def neighbors_any(mask: np.ndarray) -> np.ndarray:
    """True where at least one of the 8 neighbors is True. Edges do not wrap
    north-south; they wrap east-west because the map is a full circle."""
    padded = np.zeros((mask.shape[0] + 2, mask.shape[1] + 2), dtype=bool)
    padded[1:-1, 1:-1] = mask
    padded[1:-1, 0] = mask[:, -1]
    padded[1:-1, -1] = mask[:, 0]
    out = np.zeros_like(mask)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy or dx:
                out |= padded[1 + dy : 1 + dy + mask.shape[0], 1 + dx : 1 + dx + mask.shape[1]]
    return out


def neighbor_count(mask: np.ndarray) -> np.ndarray:
    """Number of True pixels among the 8 neighbors, with east-west wrap."""
    padded = np.zeros((mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.int16)
    padded[1:-1, 1:-1] = mask
    padded[1:-1, 0] = mask[:, -1]
    padded[1:-1, -1] = mask[:, 0]
    out = np.zeros(mask.shape, dtype=np.int16)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy or dx:
                out += padded[1 + dy : 1 + dy + mask.shape[0], 1 + dx : 1 + dx + mask.shape[1]]
    return out


def distance_to_coast(land: np.ndarray) -> np.ndarray:
    """Chebyshev distance of every pixel to the nearest pixel of the other
    class. A pixel next to the other class has distance 1."""
    dist = np.zeros(land.shape, dtype=np.int32)
    reached = np.zeros(land.shape, dtype=bool)
    frontier = (land & neighbors_any(~land)) | (~land & neighbors_any(land))
    step = 1
    while not reached.all():
        new = frontier & ~reached
        if not new.any():
            dist[~reached] = step  # only one class on the whole map
            break
        dist[new] = step
        reached |= new
        frontier = neighbors_any(reached)
        step += 1
    return dist


def coastline_length(land: np.ndarray, valid: np.ndarray) -> int:
    """Number of edges between a land pixel and a sea pixel, both valid.
    Counts the 4-neighborhood, with east-west wrap."""
    v = valid
    horizontal = (land[:, :-1] != land[:, 1:]) & v[:, :-1] & v[:, 1:]
    wrap = (land[:, -1] != land[:, 0]) & v[:, -1] & v[:, 0]
    vertical = (land[:-1, :] != land[1:, :]) & v[:-1, :] & v[1:, :]
    return int(horizontal.sum() + wrap.sum() + vertical.sum())


def isolated_share(land: np.ndarray, valid: np.ndarray) -> float:
    """Share of valid pixels that disagree with at least 6 of their 8 neighbors."""
    same = np.where(land, neighbor_count(land & valid), neighbor_count(~land & valid))
    other = np.where(land, neighbor_count(~land & valid), neighbor_count(land & valid))
    isolated = valid & (other >= 6) & (same <= 2)
    return float(isolated.sum() / max(valid.sum(), 1))


def row_weights(height: int) -> np.ndarray:
    """Area weight of each pixel row: cosine of the latitude of its center."""
    lats = np.array([mapgrid.pixel_to_lonlat(0, y, 1, height)[1] for y in range(height)])
    return np.cos(np.radians(lats))


# ---------------------------------------------------------------- metrics ----

def classification_metrics(pred_land: np.ndarray, pred_sea: np.ndarray, true_land: np.ndarray,
                           asked: np.ndarray, w: np.ndarray) -> dict[str, float]:
    """All scores on the asked pixels, with per-pixel weights w."""
    true_sea = ~true_land
    tp = float((w * (pred_land & true_land & asked)).sum())   # land called land
    fp = float((w * (pred_land & true_sea & asked)).sum())    # sea called land
    fn = float((w * (pred_sea & true_land & asked)).sum())    # land called sea
    tn = float((w * (pred_sea & true_sea & asked)).sum())     # sea called sea
    unk_land = float((w * (~pred_land & ~pred_sea & true_land & asked)).sum())
    unk_sea = float((w * (~pred_land & ~pred_sea & true_sea & asked)).sum())
    n_land = tp + fn + unk_land
    n_sea = tn + fp + unk_sea
    answered = tp + fp + fn + tn
    total = answered + unk_land + unk_sea

    def div(a: float, b: float) -> float:
        return a / b if b else float("nan")

    land_recall = div(tp, n_land)
    sea_recall = div(tn, n_sea)
    land_precision = div(tp, tp + fp)
    sea_precision = div(tn, tn + fn)
    accuracy = div(tp + tn, total)
    po = accuracy
    pe = div((tp + fp) * n_land + (tn + fn) * n_sea, total * total)
    kappa = div(po - pe, 1 - pe)
    mcc_den = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = div(tp * tn - fp * fn, mcc_den)
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn, "unknown": unk_land + unk_sea, "asked": total,
        "land_share": div(tp + fp, answered),
        "true_land_share": div(n_land, total),
        "iou_land": div(tp, tp + fp + fn + unk_land),
        "iou_sea": div(tn, tn + fn + fp + unk_sea),
        "accuracy": accuracy,
        "balanced_accuracy": (land_recall + sea_recall) / 2,
        "land_recall": land_recall, "sea_recall": sea_recall,
        "land_precision": land_precision, "sea_precision": sea_precision,
        "land_f1": div(2 * tp, 2 * tp + fp + fn + unk_land),
        "sea_f1": div(2 * tn, 2 * tn + fn + fp + unk_sea),
        "kappa": kappa, "mcc": mcc,
    }


def pct(x: float) -> str:
    return "   nan" if math.isnan(x) else f"{x * 100:6.1f}"


def report(name: str, img: np.ndarray, ref_land: np.ndarray, colors: mapgrid.Colors, ref_dist: np.ndarray,
           ref_coast: int, ref_isolated: float) -> None:
    pred_land = mapgrid.mask_of(img, colors.land)
    pred_sea = mapgrid.mask_of(img, colors.sea)
    asked = ~mapgrid.mask_of(img, colors.blank)
    h, w_ = img.shape[:2]
    ones = np.ones((h, w_))
    area = np.repeat(row_weights(h)[:, None], w_, axis=1)
    m = classification_metrics(pred_land, pred_sea, ref_land, asked, ones)
    a = classification_metrics(pred_land, pred_sea, ref_land, asked, area)

    print(f"\n{name}")
    print(f"  asked {int(m['asked'])} of {h * w_}  answered {int(m['asked'] - m['unknown'])}  unknown {int(m['unknown'])}")
    print(f"  confusion (true -> said): land->land {int(m['tp'])}  land->sea {int(m['fn'])}  "
          f"sea->sea {int(m['tn'])}  sea->land {int(m['fp'])}")
    print(f"  land share of answers {pct(m['land_share'])}%   true land share {pct(m['true_land_share'])}%")
    print(f"  {'':22s}{'pixels':>8s}{'area':>8s}")
    rows = [
        ("IoU land", "iou_land"), ("IoU sea", "iou_sea"),
        ("accuracy", "accuracy"), ("balanced accuracy", "balanced_accuracy"),
        ("land recall", "land_recall"), ("land precision", "land_precision"), ("land F1", "land_f1"),
        ("sea recall", "sea_recall"), ("sea precision", "sea_precision"), ("sea F1", "sea_f1"),
        ("Cohen's kappa", "kappa"), ("Matthews corr.", "mcc"),
    ]
    for label, key in rows:
        print(f"  {label:22s}{pct(m[key]):>8s}{pct(a[key]):>8s}")
    mean_iou = (m["iou_land"] + m["iou_sea"]) / 2
    mean_iou_a = (a["iou_land"] + a["iou_sea"]) / 2
    print(f"  {'mean IoU':22s}{pct(mean_iou):>8s}{pct(mean_iou_a):>8s}")

    # accuracy by distance from the true coast
    correct = (pred_land & ref_land) | (pred_sea & ~ref_land)
    wrong = asked & ~correct
    print("  accuracy by distance from the true coast (pixels):")
    for lo, hi in DISTANCE_BINS:
        sel = asked & (ref_dist >= lo) & (ref_dist <= hi)
        n = int(sel.sum())
        if not n:
            continue
        label = f"{lo}" if lo == hi else (f"{lo}-{hi}" if hi < 10_000 else f"{lo}+")
        print(f"    {label:>6s}: {pct(correct[sel].mean())}%  of {n:5d} pixels, {int(wrong[sel].sum()):5d} wrong")
    if wrong.any():
        d = ref_dist[wrong]
        print(f"  wrong pixels: median distance to coast {np.median(d):.0f}, "
              f"{(d <= 2).mean() * 100:.0f}% within 2, {(d >= 9).mean() * 100:.0f}% at 9 or more")

    # noise
    answered = pred_land | pred_sea
    print(f"  coastline length {coastline_length(pred_land, answered)} edges (reference {ref_coast}), "
          f"isolated pixels {isolated_share(pred_land, answered) * 100:.1f}% (reference {ref_isolated * 100:.1f}%)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=Path("config.toml"))
    parser.add_argument("--reference", type=Path, help="reference map (default: basemap from the config)")
    parser.add_argument("maps", nargs="+", type=Path, help="maps to score")
    args = parser.parse_args()

    with open(args.config, "rb") as fh:
        cfg = tomllib.load(fh)
    try:
        colors = mapgrid.colors_from_config(cfg)
    except ValueError as exc:
        sys.exit(str(exc))
    ref_path = args.reference or Path(cfg.get("basemap", "maps/basemap.png"))
    if not ref_path.exists():
        sys.exit(f"reference map {ref_path} not found")
    ref = mapgrid.load_map(ref_path)
    ref_land = mapgrid.mask_of(ref, colors.land)
    ref_sea = mapgrid.mask_of(ref, colors.sea)
    if not (ref_land | ref_sea).all():
        sys.exit(f"reference map {ref_path} has pixels that are neither land nor sea")
    ref_dist = distance_to_coast(ref_land)
    ref_coast = coastline_length(ref_land, np.ones(ref_land.shape, dtype=bool))
    ref_isolated = isolated_share(ref_land, np.ones(ref_land.shape, dtype=bool))
    print(f"reference {ref_path}: land {int(ref_land.sum())}  sea {int(ref_sea.sum())}  "
          f"land share {ref_land.mean() * 100:.1f}% of pixels, {float((row_weights(ref.shape[0])[:, None] * ref_land).sum() / (row_weights(ref.shape[0]).sum() * ref.shape[1])) * 100:.1f}% of area")

    for path in args.maps:
        if not path.exists():
            print(f"\n{path}: not found")
            continue
        img = mapgrid.load_map(path)
        if img.shape != ref.shape:
            print(f"\n{path}: size {img.shape[1]}x{img.shape[0]} differs from the reference, skipped")
            continue
        report(str(path), img, ref_land, colors, ref_dist, ref_coast, ref_isolated)


if __name__ == "__main__":
    main()
