# CLAUDE.md

Guidance for agents working in this repository.

## What this project is

An experiment that draws a 320 x 160 land/sea map of the world twice: once
from Natural Earth polygons (the reference), and once by asking an LLM, one
pixel at a time, whether a given latitude and longitude is land or sea.

## Layout

- `mapgrid.py` — shared constants and helpers: grid size, the `Colors` loader
  and grayscale check for the `[colors]` table, pixel center to lon/lat,
  atomic PNG save, model name sanitizer. Change grid geometry here and
  nowhere else.
- `make_basemap.py` — rasterizes `data/ne_110m_land/*.shp` to `maps/basemap.png`.
  Reads the colors from `config.toml` (`--config` to point elsewhere).
- `llm_map.py` — the pixel loop. Writes `maps/llm_<sanitized model name>.png`.
- `score_map.py` — compares maps with a reference (the base map by default,
  or any map with `--reference`): IoU per class, accuracy, balanced
  accuracy, precision, recall, F1, kappa, MCC, each by pixel count and
  area-weighted by cos(latitude); accuracy by distance from the true coast
  (fuzziness versus ignorance); coastline length and isolated-pixel share
  (noise). Blank pixels are left out, unknown pixels count as wrong.
- `make_composite.py` — draws the base map and every `maps/llm_*.png` as
  tiles in one image with matplotlib, the model name as the only subtitle
  per tile, no overall title and no scores. Writes `composite.png` in the
  repository root.
- `config.toml` — the four pixel colors, the prompt file name, endpoints,
  models, request parameters, save interval. Colors are defined here and
  nowhere else; there are no defaults in code. Comments in the file stay
  minimal; the reasons behind the values live in this document.
- `prompt.txt` — the whole prompt, nothing else, the same for every model.
  A Python format string with `{lat}` and `{lon}` (and optionally `{lat_abs}`,
  `{lon_abs}`, `{ns}`, `{ew}` for the "23.438° N" style, and `{x}`, `{y}`).
  The fields are built by `prompt_fields` in `llm_map.py`, which checks at
  start that the template formats and names both coordinates.
- `download_data.sh` — fetches the Natural Earth shapefile into `data/`.
- `AGENTS.md` — symlink to this file.
- `data/` — downloaded Natural Earth files. Untracked, treat as read-only input.
- `maps/` — generated output. Untracked.

## Environment

Python 3.14 venv at `.venv/`. Always run scripts with `.venv/bin/python`.
Install with `.venv/bin/pip install -r requirements.txt`.

    .venv/bin/python make_basemap.py [--config FILE] [--all-touched]
    .venv/bin/python llm_map.py [--config FILE] [--model NAME] [--order random] [--limit N] [--save-interval S] [--retry-unknown]
    .venv/bin/python score_map.py [--config FILE] [--reference MAP] maps/llm_*.png
    .venv/bin/python make_composite.py [--config FILE] [--out composite.png] [MAP ...]

## Invariants that must not break

- **The output PNG is the run state.** White = not asked, gray = no usable
  answer, light brick = land, deep blue = sea. The exact RGB values from the
  `[colors]` table in `config.toml` are compared byte for byte on resume, so
  changing a color orphans every existing `maps/llm_*.png`. Do not add
  dithering, alpha, palette conversion or any color change in code.
- **The four colors are far apart in grayscale on purpose** (BT.601, the
  formula Pillow uses for "L" images: blank 255, land 186, unknown 115,
  sea 42), for low vision users and black-and-white output.
  `mapgrid.colors_from_config` rejects a table where two colors are closer
  than `MIN_GRAY_GAP` (32) steps. Keep that check. The current colors are
  light brick for land and deep blue for sea. To change a color, delete
  the existing LLM maps or recolor them first.
- **Saves are atomic.** Always go through `mapgrid.save_map_atomic` (temp
  file in the same directory, fsync, `os.replace`). Never write the output
  path directly.
- **CTRL-C is handled on purpose.** First SIGINT sets a flag, the request in
  flight completes, the map is saved, exit code 130. Second SIGINT raises
  KeyboardInterrupt and exits at once. Keep the `finally` block saving.
- **Every pixel is a fresh chat**: one user message built from `prompt.txt`,
  no system prompt, no history. Do not batch several pixels into one request
  without being asked.
- **The grid formula is fixed**: pixel (x, y) center is
  lon = -180 + (x + 0.5) * 360 / W, lat = 90 - (y + 0.5) * 180 / H.
  The base map uses pixel-center sampling for the same reason.

## Endpoint facts (measured on Ollama 0.40)

- The endpoint is Ollama's OpenAI-compatible API at `http://mac:11434/v1`.
  The host `mac` is this machine.
- All three configured models (Qwen3.8, Gemma 4, Ornith 1.5) think by
  default. On this server only `reasoning_effort = "none"` turns thinking
  off. `think = false`, `chat_template_kwargs = { enable_thinking = false }`
  and the named levels are ignored or still think. Per model:
  - Qwen3.8: the model card documents `chat_template_kwargs` with
    `enable_thinking = false` and `reasoning_effort` low, medium, xhigh.
    Through Ollama's OpenAI endpoint the first is ignored and the three
    levels all think for about 180 tokens, about 6 s per pixel.
  - Gemma 4: `think = false`, the `chat_template_kwargs` form and
    `reasoning_effort = "low"` are all ignored and it still thinks, 150 to
    250 tokens, about 7 s per pixel.
  - Ornith 1.5: `/api/show` has no thinking metadata for this GGUF import,
    but its template honors `enable_thinking`. With thinking on it reasons
    for about 60 tokens on a test point.
- `max_tokens = 4000` is large on purpose so that a run with thinking on
  has room for the reasoning before the one-word answer. With thinking
  off the answer is 1 or 2 tokens and the limit never matters.
- Thinking off, full maps with the current prompt: Ornith 1.5 2h36m
  (5.4 px/s), Qwen3.8 7h33m (2.2 px/s), Gemma 4 9h11m (1.5 px/s).
  Thinking on: about 6 s per pixel, roughly 3.5 days, much more accurate.
- To run every model unattended, start a detached script (`nohup`) that
  runs `llm_map.py --model` for one model after another and calls
  `ollama stop` after each. It survives the end of the agent session; the
  maps are the state, so a rerun of the same command resumes.
- On this Mac, prompt evaluation dominates the time per pixel and the single
  answer token costs a fixed 0.27 s on Gemma 4. Ollama reuses the cached
  prefix of the previous request, so a prompt with the coordinates in the
  first line evaluates slower than one with the coordinates last (1.07 vs
  0.64 s per pixel on Gemma 4 for prompts of about 100 tokens). The current
  55-token prompt is as fast as a cached one. Prefer a cache-friendly layout
  when the accuracy difference is small; this is a judgement call.
- Background tasks on the machine change the time per pixel from one run to
  the next, so compare timings only from interleaved requests.
- The machine runs one model at a time. Before switching models, unload the
  current one with `ollama stop <model>`; do not rely on Ollama to swap.
  `ollama ps` shows what is loaded. Two runs on different models at the same
  time make Ollama reload a model on every request.
- Anything under `[endpoints.models.params]` is passed through to the
  request body unchanged via `extra_body`.
- To see which thinking values a model accepts, ask Ollama's own API:
  `curl http://mac:11434/api/show -d '{"model":"NAME"}'`. The `thinking`
  field lists the accepted `think` values and the default. Through the
  OpenAI endpoint `reasoning_effort = "none"` maps to `false` and the level
  names map to the named values. The field is null for some GGUF imports;
  then read the template.
- Before adding a model, verify that thinking is really off, not only
  hidden: with `reasoning_effort = "none"` the `usage.completion_tokens`
  must be 1 or 2 for the one-word answer, and the answer must survive
  `max_tokens = 5`. Muse Glimmer failed this (about 250 hidden tokens per
  pixel, 11 s) and was dropped.

## Prompt rules and what was measured

The prompt is a knowledge test. It must not lead the model: no hints about
the answer, not even general ones such as the land/water ratio of Earth,
and no instruction on what to answer when unsure. One prompt is used for
all models; a variant that helps one model is documented here, not
configured.

Measured on the same 800 random pixels per model with thinking off,
agreement with the base map (the base map is 33% land):

| prompt | Gemma 4 | Ornith 1.5 | Qwen3.8 |
|---|---|---|---|
| first round: "geography oracle" persona, definitions, signed decimals | 66.0 | 60.6 | 38.5 |
| same, coordinates as "23.438° S, 45.562° W" | 70.5 | 56.6 | 62.4 |
| question with coordinates first, then definitions and answer rule | 76.2 | 76.1 | 39.4 |
| definitions and answer rule first, question with coordinates last | 73.6 | 76.4 | 47.4 |
| definitions and answer rule, coordinates alone on the last line | 67.5 | 72.6 | 36.1 |
| current `prompt.txt`: one-line question, "Answer with one word" | 71.1 | 75.8 | 67.8 |

Observations:

- Cardinal coordinates beat signed decimals for Gemma 4 and Qwen3.8.
- Dropping the persona and asking a plain question helps Gemma 4 and
  Ornith. Qwen3.8 answers land for about 90% of pixels on every question
  form that also carries the definitions line, but not on the bare question.
- With the bare question Ornith undercalls land (20% of answers); with the
  definitions line it is balanced (39%).
- The models favor the option named last, so keep "land or sea" in that
  order. Few-shot examples did not help.
- Hints were tried and rejected on principle, but for the record: "If you
  are not sure, answer sea" raised every model by 2 to 5 points, and
  stating that 71% of Earth is water made the models say land more often.

## Results of the second round (current prompt, full maps)

| | Ornith 1.5 | Gemma 4 | Qwen3.8 |
|---|---|---|---|
| agreement with the base map | 76.2% | 70.9% | 67.9% |
| first round | 61.0% | 66.1% | 38.7% |
| land share of answers (true 33%) | 20% | 58% | 55% |
| mean IoU (pixels / area) | 55 / 63 | 55 / 54 | 51 / 54 |
| Cohen's kappa (pixels / area) | 41 / 52 | 45 / 43 | 38 / 42 |

- The full-map result of each model matched its 800-pixel trial within
  half a point.
- Ornith undercalls land: it is the only map where the continents are
  recognizable, but it drops Antarctica and thins the tropics. Its high
  agreement comes partly from answering sea; by IoU and kappa it is not
  clearly ahead.
- Gemma 4 and Qwen3.8 overcall land. The oceans show as blobs in the right
  places and the continents are smeared out to sea. Qwen3.8 calls almost
  the whole Arctic Ocean land. Both show faint diagonal streaks where
  latitude and longitude values coincide, a sign of reacting to the
  numbers rather than to the place.
- All three are near coin-flip on the coast itself and reach about 86%
  at 9 to 16 pixels from it. Beyond 17 pixels Ornith and Gemma 4 are 97 to
  99% right (fuzz around known shapes); Qwen3.8 is 86% (ignorance of the
  shapes). All three have a land/sea boundary 2.5 times longer than the
  true coast: per-pixel noise from fresh chats with no context.
- Ornith left 40 pixels unknown, all on ice: the sea north and east of
  Greenland, Greenland itself, the Antarctic coast, and one in Lake
  Superior. It answers "Antarctica" (also at 88° N) or "Greenland"
  instead of a word from the allowed pair. At temperature 0 this repeats,
  so `--retry-unknown` changes nothing. Gray is the honest color for
  these; do not teach the parser to map place names to land or sea.

## Testing without the real model

A fake OpenAI server lives only in the session scratchpad, not in the repo.
To test resume and CTRL-C behavior, write a small `http.server` that returns
`{"choices":[{"message":{"content":"sea"}}]}` and point a throwaway config
at it. The throwaway config needs `out_dir` outside `maps/`, a `[colors]`
table (copy the real one) and a `prompt_file` (the real `prompt.txt` works).
Use `--limit` and `--save-interval 1` to keep test runs short.

## Comparing prompts

The prompt trial harness is not in the repo; write it in the scratchpad.
What worked: pick a fixed random sample of 800 pixels with a seeded RNG,
ask the real model with the candidate prompt and the model's params from
`config.toml`, and score the answers against `maps/basemap.png` (accuracy,
land recall, sea recall, share of answers that say land). 800 pixels give a
standard error of about 1.5 points and take 3 to 9 minutes per prompt and
model. Run one model at a time and unload it before the next (see above).
Run the first-round prompt first as a check: its sample result must match
the full map within half a point.

## Conventions

- Keep scripts dependency-light: numpy, Pillow, rasterio, pyshp, openai,
  matplotlib (only for `make_composite.py`), and the standard library
  (`tomllib`, `argparse`, `signal`).
- Do not run git commands unless asked. Do not commit `data/` or `maps/`.
- Deleting `maps/llm_<model>.png` is the way to restart a model from scratch.
  Say so instead of adding a `--reset` flag.
- Write explanations and documentation in Simplified Technical English where
  it reads naturally: short sentences, one idea each, active voice.
