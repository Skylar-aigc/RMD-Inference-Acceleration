"""Build a reproducible, text-only 2K training subset from VidProM.

Only Hugging Face dataset-viewer rows are downloaded; no generated videos are
fetched. The resulting file is shuffled deterministically and excludes exact
matches from the official VBench prompt suite.
"""
from __future__ import annotations

import argparse
import html
import json
import random
import re
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

DATASET = "WenhaoWang/VidProM"
CONFIG = "VidProM_unique"
SPLIT = "train"
TOTAL_ROWS = 1_672_454
ROWS_URL = "https://datasets-server.huggingface.co/rows"
VBENCH_URL = (
    "https://raw.githubusercontent.com/Vchitect/VBench/master/"
    "vbench/VBench_full_info.json"
)
SAFETY_FIELDS = (
    "toxicity",
    "obscene",
    "identity_attack",
    "insult",
    "threat",
    "sexual_explicit",
)
BLOCKED_TERMS = re.compile(
    r"\b(?:nsfw|nude|nudity|porn|pornographic|sexual|sexually|gore|"
    r"beheading|suicide)\b",
    re.IGNORECASE,
)
PLATFORM_TAIL = re.compile(
    r"(?:\s+|[,;]\s*|(?=--))(?:-{1,2}\s*(?:ar|fps|ftp|motion|seed|v|q|"
    r"g|style|chaos|iw|video|no|neg(?:ative)?|gs|hide|camera|zoom|sharp)"
    r"(?=[\s:=,-]|\d|$)|ar--)"
    r".*$",
    re.IGNORECASE,
)
MESSAGE_TAIL = re.compile(r"\s*Message:.*$", re.IGNORECASE)


def _get_json(url: str, retries: int = 5) -> object:
    request = urllib.request.Request(url, headers={"User-Agent": "RMD-dataset-builder/1.0"})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return json.load(response)
        except urllib.error.HTTPError as exc:
            if attempt == retries - 1:
                raise
            retry_after = exc.headers.get("Retry-After") if exc.code == 429 else None
            time.sleep(float(retry_after) if retry_after else 5 * 2**attempt)
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(2**attempt)
    raise RuntimeError("unreachable")


def _normalize_for_match(text: str) -> str:
    text = unicodedata.normalize("NFKC", html.unescape(text)).casefold()
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def _load_vbench_prompts() -> set[str]:
    records = _get_json(VBENCH_URL)
    return {
        _normalize_for_match(record["prompt_en"])
        for record in records
        if isinstance(record, dict) and record.get("prompt_en")
    }


def _clean_prompt(row: dict, safety_threshold: float, vbench: set[str]) -> str | None:
    if any(float(row.get(field, 1.0)) > safety_threshold for field in SAFETY_FIELDS):
        return None

    prompt = unicodedata.normalize("NFKC", html.unescape(str(row.get("prompt", ""))))
    prompt = re.sub(r"^\s*/imagine\s+prompt:\s*", "", prompt, flags=re.IGNORECASE)
    prompt = MESSAGE_TAIL.sub("", prompt)
    prompt = PLATFORM_TAIL.sub("", prompt)
    prompt = re.sub(r"\s+", " ", prompt).strip(" ,.;:-")

    if "_" in prompt or re.search(r"(?:image|prompt):\S", prompt, re.IGNORECASE):
        return None
    if not 24 <= len(prompt) <= 420:
        return None
    words = re.findall(r"[A-Za-z]+(?:['-][A-Za-z]+)?", prompt)
    if not 5 <= len(words) <= 75:
        return None
    if len("".join(words)) / max(len(prompt), 1) < 0.62:
        return None
    if "http://" in prompt.casefold() or "https://" in prompt.casefold():
        return None
    if "attachment" in prompt.casefold() or "message:" in prompt.casefold():
        return None
    if PLATFORM_TAIL.search(prompt) or BLOCKED_TERMS.search(prompt):
        return None
    if _normalize_for_match(prompt) in vbench:
        return None
    return prompt


def _fetch_rows(offset: int, length: int = 100) -> list[dict]:
    query = urllib.parse.urlencode(
        {
            "dataset": DATASET,
            "config": CONFIG,
            "split": SPLIT,
            "offset": offset,
            "length": length,
        }
    )
    payload = _get_json(f"{ROWS_URL}?{query}")
    return [item["row"] for item in payload["rows"]]


def build_subset(count: int, seed: int, safety_threshold: float) -> list[str]:
    rng = random.Random(seed)
    vbench = _load_vbench_prompts()
    offsets = list(range(0, TOTAL_ROWS - 100, 100))
    rng.shuffle(offsets)

    selected: dict[str, str] = {}
    for block_number, offset in enumerate(offsets, start=1):
        for row in _fetch_rows(offset):
            prompt = _clean_prompt(row, safety_threshold, vbench)
            if prompt is not None:
                selected.setdefault(_normalize_for_match(prompt), prompt)
        if len(selected) >= count:
            break
        if block_number >= 500:
            raise RuntimeError(f"Only found {len(selected)} eligible prompts after 500 blocks")

    prompts = list(selected.values())
    rng.shuffle(prompts)
    return prompts[:count]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--safety-threshold", type=float, default=0.05)
    parser.add_argument("--output", default="examples/vidprom_prompts_2000.txt")
    args = parser.parse_args()

    prompts = build_subset(args.count, args.seed, args.safety_threshold)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(prompts) + "\n", encoding="utf-8")
    print(f"Wrote {len(prompts)} prompts to {output}")


if __name__ == "__main__":
    main()
