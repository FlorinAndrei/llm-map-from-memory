#!/usr/bin/env python
"""Ask an LLM, pixel by pixel, whether each point of the world map is land or sea.

The output image doubles as the run state: white pixels are not answered yet,
so a run that stops for any reason continues where it left off when started
again with the same model. The file is saved atomically at a fixed time interval and
on exit. The first CTRL-C finishes the request in flight, saves, and exits.
A second CTRL-C exits at once; the answers since the last save are then lost
but the file on disk stays valid.

Every pixel is a fresh chat: one user message built from the prompt file,
no system prompt, no history from earlier pixels.
"""

from __future__ import annotations

import argparse
import re
import signal
import sys
import time
import tomllib
from pathlib import Path
from typing import Any

import numpy as np
from openai import OpenAI

import mapgrid

ANSWER_RE = re.compile(r"\b(land|sea)\b", re.IGNORECASE)
THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


class StopRequested(Exception):
    """Raised by the loop when the user asked to stop."""


# ---------------------------------------------------------------- config ----

def load_config(path: Path) -> dict[str, Any]:
    with open(path, "rb") as fh:
        cfg = tomllib.load(fh)
    cfg.setdefault("width", mapgrid.WIDTH)
    cfg.setdefault("height", mapgrid.HEIGHT)
    cfg.setdefault("out_dir", "maps")
    cfg.setdefault("save_interval", 60.0)
    cfg.setdefault("max_consecutive_errors", 10)
    cfg.setdefault("prompt_file", "prompt.txt")
    cfg["prompt"] = load_prompt(Path(cfg["prompt_file"]))
    try:
        cfg["colors"] = mapgrid.colors_from_config(cfg)
    except ValueError as exc:
        sys.exit(str(exc))
    if not cfg.get("endpoints"):
        sys.exit("config: at least one [[endpoints]] entry is required")
    for ep in cfg["endpoints"]:
        if "url" not in ep:
            sys.exit("config: every endpoint needs a url")
        if not ep.get("models"):
            sys.exit(f"config: endpoint {ep['url']} needs at least one [[endpoints.models]] entry")
        for model in ep["models"]:
            if "name" not in model:
                sys.exit(f"config: a model of endpoint {ep['url']} has no name")
            model.setdefault("params", {})
    return cfg


def prompt_fields(lat: float, lon: float, x: int, y: int) -> dict[str, Any]:
    """The values a prompt template can use.

    lat, lon: signed decimal degrees.  lat_abs, lon_abs: without sign.
    ns, ew: hemisphere letters N/S and E/W.  x, y: pixel column and row.
    """
    return {
        "lat": lat, "lon": lon,
        "lat_abs": abs(lat), "lon_abs": abs(lon),
        "ns": "N" if lat >= 0 else "S",
        "ew": "E" if lon >= 0 else "W",
        "x": x, "y": y,
    }


def load_prompt(path: Path) -> str:
    """Read the prompt template and check that it formats with the available fields."""
    try:
        prompt = path.read_text(encoding="utf-8")
    except OSError as exc:
        sys.exit(f"config: cannot read prompt_file {path}: {exc}")
    if not prompt.strip():
        sys.exit(f"config: prompt_file {path} is empty")
    try:
        prompt.format(**prompt_fields(0.0, 0.0, 0, 0))
    except (KeyError, ValueError, IndexError) as exc:
        sys.exit(f"config: prompt_file {path} is not a valid format string "
                 f"(fields: lat, lon, lat_abs, lon_abs, ns, ew, x, y; "
                 f"use {{{{ and }}}} for literal braces): {exc}")
    if "{lat" not in prompt or "{lon" not in prompt:
        sys.exit(f"config: prompt_file {path} must contain {{lat}} and {{lon}}")
    return prompt


def select_model(cfg: dict[str, Any], endpoint_url: str | None, model_name: str | None) -> tuple[dict, dict]:
    """Return (endpoint, model) for the requested names, or the first ones."""
    candidates = [
        (ep, m)
        for ep in cfg["endpoints"]
        for m in ep["models"]
        if (endpoint_url is None or ep["url"] == endpoint_url)
        and (model_name is None or m["name"] == model_name)
    ]
    if not candidates:
        sys.exit(f"config: no model matches endpoint={endpoint_url!r} model={model_name!r}")
    return candidates[0]


# ------------------------------------------------------------------- llm ----

def parse_answer(text: str | None) -> str | None:
    """Return 'land', 'sea' or None. The last match wins, so a short chain of
    reasoning that ends with the answer still parses."""
    if not text:
        return None
    text = THINK_RE.sub(" ", text)
    matches = ANSWER_RE.findall(text)
    return matches[-1].lower() if matches else None


def ask(client: OpenAI, model: dict[str, Any], prompt: str) -> str | None:
    """Run one fresh chat with a single user message and return the parsed answer, or None."""
    response = client.chat.completions.create(
        model=model["name"],
        messages=[{"role": "user", "content": prompt}],
        extra_body=model["params"],
    )
    message = response.choices[0].message
    return parse_answer(message.content)


# ------------------------------------------------------------------ main ----

def format_eta(seconds: float) -> str:
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m{s:02d}s"


def land_share(img: np.ndarray, colors: mapgrid.Colors) -> str:
    """Share of land among the answered pixels of the whole map, as text."""
    land = mapgrid.mask_of(img, colors.land).sum()
    sea = mapgrid.mask_of(img, colors.sea).sum()
    answered = land + sea
    if not answered:
        return "no answers yet"
    return f"land {land / answered * 100:.1f}% of {answered} answered"


def report(img: np.ndarray, colors: mapgrid.Colors, basemap_path: str | None) -> None:
    land = mapgrid.mask_of(img, colors.land)
    sea = mapgrid.mask_of(img, colors.sea)
    unknown = mapgrid.mask_of(img, colors.unknown)
    blank = mapgrid.mask_of(img, colors.blank)
    total = img.shape[0] * img.shape[1]
    print(f"pixels: land {land.sum()}  sea {sea.sum()}  unknown {unknown.sum()}  blank {blank.sum()}  of {total}")
    if not basemap_path or not Path(basemap_path).exists():
        return
    base = mapgrid.load_map(basemap_path)
    if base.shape != img.shape:
        print(f"basemap {basemap_path} has a different size, agreement not computed")
        return
    answered = land | sea
    if not answered.any():
        return
    base_land = mapgrid.mask_of(base, colors.land)
    agree = (land == base_land)[answered].mean() * 100.0
    land_recall = (land & base_land).sum() / max((answered & base_land).sum(), 1) * 100.0
    sea_recall = (sea & ~base_land).sum() / max((answered & ~base_land).sum(), 1) * 100.0
    print(f"agreement with basemap on answered pixels: {agree:.1f}%  (land correct {land_recall:.1f}%, sea correct {sea_recall:.1f}%)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=Path("config.toml"))
    parser.add_argument("--endpoint", help="endpoint url from the config (default: the first one)")
    parser.add_argument("--model", help="model name from the config (default: the first one)")
    parser.add_argument("--retry-unknown", action="store_true", help="ask again for pixels marked unknown")
    parser.add_argument("--limit", type=int, help="stop after this many new answers (for testing)")
    parser.add_argument("--save-interval", type=float, help="seconds between saves, overrides save_interval from the config")
    parser.add_argument("--order", choices=["rows", "random"], default="rows",
                        help="visit pending pixels row by row, or in random order so a partial map samples the whole globe")
    args = parser.parse_args()

    cfg = load_config(args.config)
    endpoint, model = select_model(cfg, args.endpoint, args.model)
    width, height = cfg["width"], cfg["height"]
    colors: mapgrid.Colors = cfg["colors"]
    save_interval = args.save_interval if args.save_interval is not None else float(cfg["save_interval"])

    out_path = Path(cfg["out_dir"]) / f"llm_{mapgrid.sanitize_name(model['name'])}.png"
    if out_path.exists():
        img = mapgrid.load_map(out_path)
        if img.shape != (height, width, 3):
            sys.exit(f"{out_path} is {img.shape[1]}x{img.shape[0]}, config says {width}x{height}")
        print(f"resuming {out_path}")
    else:
        img = mapgrid.new_blank(colors.blank, width, height)
        mapgrid.save_map_atomic(img, out_path)
        print(f"created {out_path}")

    pending = mapgrid.mask_of(img, colors.blank)
    if args.retry_unknown:
        pending |= mapgrid.mask_of(img, colors.unknown)
    todo = list(zip(*np.nonzero(pending)))  # (y, x) in row-major order
    if args.order == "random":
        np.random.default_rng().shuffle(todo)
    total = width * height
    print(f"model {model['name']} at {endpoint['url']}, prompt {cfg['prompt_file']}: {len(todo)} of {total} pixels to do")
    if not todo:
        report(img, colors, cfg.get("basemap"))
        return

    client = OpenAI(
        base_url=endpoint["url"],
        api_key=endpoint.get("api_key") or "none",
        timeout=endpoint.get("timeout", 120),
        max_retries=endpoint.get("max_retries", 2),
    )

    stop = {"requested": False}
    original_sigint = signal.getsignal(signal.SIGINT)

    def on_sigint(signum, frame):
        if stop["requested"]:
            signal.signal(signal.SIGINT, original_sigint)
            raise KeyboardInterrupt
        stop["requested"] = True
        print("\nstop requested, finishing the current pixel and saving (CTRL-C again to quit now)", flush=True)

    signal.signal(signal.SIGINT, on_sigint)

    done = 0
    unsaved = 0
    errors_in_a_row = 0
    interval = {"land": 0, "sea": 0, "unknown": 0}    # answers since the last progress line
    t0 = time.monotonic()
    last_save = t0
    exit_code = 0

    def save() -> None:
        nonlocal unsaved, last_save
        mapgrid.save_map_atomic(img, out_path)
        unsaved = 0
        last_save = time.monotonic()

    try:
        for y, x in todo:
            if stop["requested"]:
                raise StopRequested
            if args.limit is not None and done >= args.limit:
                break
            lon, lat = mapgrid.pixel_to_lonlat(int(x), int(y), width, height)
            prompt = cfg["prompt"].format(**prompt_fields(lat, lon, int(x), int(y)))
            try:
                answer = ask(client, model, prompt)
            except Exception as exc:  # network or API failure: leave the pixel blank
                errors_in_a_row += 1
                print(f"error at ({x},{y}) lat {lat:.3f} lon {lon:.3f}: {type(exc).__name__}: {exc}", flush=True)
                if errors_in_a_row >= cfg["max_consecutive_errors"]:
                    print("too many consecutive errors, stopping", flush=True)
                    exit_code = 1
                    break
                continue
            errors_in_a_row = 0

            if answer == "land":
                img[y, x] = colors.land
            elif answer == "sea":
                img[y, x] = colors.sea
            else:
                answer = "unknown"
                img[y, x] = colors.unknown
            interval[answer] += 1
            done += 1
            unsaved += 1

            if time.monotonic() - last_save >= save_interval:
                save()
                elapsed = time.monotonic() - t0
                rate = done / elapsed if elapsed > 0 else 0.0
                remaining = len(todo) - done
                eta = format_eta(remaining / rate) if rate > 0 else "?"
                print(
                    f"{total - remaining:6d}/{total}  {rate:5.1f} px/s  eta {eta}  "
                    f"this interval: land {interval['land']}  sea {interval['sea']}  unknown {interval['unknown']}  "
                    f"overall: {land_share(img, colors)}",
                    flush=True,
                )
                interval = {"land": 0, "sea": 0, "unknown": 0}
    except StopRequested:
        print("stopped by user")
        exit_code = 130
    except KeyboardInterrupt:
        print("\ninterrupted")
        exit_code = 130
    finally:
        signal.signal(signal.SIGINT, original_sigint)
        if unsaved:
            save()
        elapsed = time.monotonic() - t0
        print(f"this run: {done} pixels in {format_eta(elapsed)}, saved to {out_path}")
        report(img, colors, cfg.get("basemap"))

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
