"""
Pre-process hsk_all_by_length.csv into data/vocab.json for the static web app.
Run once before deploying:  python build_data.py
"""
import csv
import json
import math
import os
import re
import unicodedata
from pathlib import Path


SRC = Path(__file__).with_name("hsk_all_by_length.csv")
DST_DIR = Path(__file__).with_name("data")
DST = DST_DIR / "vocab.json"


def normalize_pinyin(text: str) -> str:
    if not text:
        return ""
    normalized = unicodedata.normalize("NFD", text.lower())
    without_tones = "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")
    clean = re.sub(r"[^a-z0-9 ]+", " ", without_tones)
    return re.sub(r"\s+", " ", clean).strip()


def parse_official_hsk(levels_text: str):
    if not levels_text or not levels_text.strip():
        return None
    matches = re.findall(r"(?:newest|new|old)-(\d+)", levels_text)
    if not matches:
        return None
    nums = sorted({int(m) for m in matches})
    return nums[0] if nums else None


def main():
    if not SRC.exists():
        raise FileNotFoundError(f"Missing: {SRC}")

    rows = []
    with open(SRC, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    # Parse frequency_rank
    freq_ranks = []
    for r in rows:
        try:
            fr = float(r.get("frequency_rank", ""))
            freq_ranks.append(fr)
        except (ValueError, TypeError):
            freq_ranks.append(None)

    max_freq = max((f for f in freq_ranks if f is not None), default=999999)

    for i, r in enumerate(rows):
        if freq_ranks[i] is None:
            freq_ranks[i] = max_freq + 100000

    # Compute proficiency_10 via quantile buckets
    indexed = sorted(range(len(rows)), key=lambda i: freq_ranks[i])
    n = len(indexed)
    prof = [0] * n
    for rank_pos, orig_idx in enumerate(indexed):
        bucket = min(int(rank_pos * 10 / n) + 1, 10)
        prof[orig_idx] = bucket

    out = []
    for i, r in enumerate(rows):
        hanzi = (r.get("hanzi_simplified") or "").strip()
        if not hanzi:
            continue
        pinyin = (r.get("pinyin") or "").strip()
        english = (r.get("english_meanings") or "").strip()
        char_count = 0
        try:
            char_count = int(r.get("character_count", 0))
        except (ValueError, TypeError):
            char_count = len(hanzi)

        hsk = parse_official_hsk(r.get("hsk_levels", ""))
        example_zh = (r.get("example_zh") or "").strip()
        example_en = (r.get("example_en") or "").strip()

        length_label = "1 char" if char_count == 1 else f"{char_count} chars" if char_count <= 3 else "4+ chars"

        out.append({
            "h": hanzi,
            "p": pinyin,
            "e": english,
            "c": char_count,
            "f": round(freq_ranks[i]),
            "l": hsk,
            "v": prof[i],
            "ll": length_label,
            "pk": normalize_pinyin(pinyin),
            "xz": example_zh,
            "xe": example_en,
        })

    DST_DIR.mkdir(exist_ok=True)
    with open(DST, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, separators=(",", ":"))

    size_kb = os.path.getsize(DST) / 1024
    print(f"Wrote {len(out)} entries to {DST} ({size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
