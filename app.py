import re
import unicodedata
import json
from difflib import SequenceMatcher
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="Mandarin Proficiency Trainer", page_icon="汉", layout="wide")

DATA_PATH = Path(__file__).with_name("hsk_all_by_length.csv")


def normalize_pinyin(text: str) -> str:
    if not isinstance(text, str):
        return ""
    normalized = unicodedata.normalize("NFD", text.lower())
    without_tones = "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")
    clean = re.sub(r"[^a-z0-9 ]+", " ", without_tones)
    clean = re.sub(r"\s+", " ", clean).strip()
    return clean


def normalize_zh_answer(text: str) -> str:
    if not isinstance(text, str):
        return ""
    cleaned = re.sub(r"[\s，。！？、,.!?;:：；\"'“”‘’()（）\[\]{}]", "", text)
    return cleaned.strip()


def normalize_en_answer(text: str) -> str:
    if not isinstance(text, str):
        return ""
    cleaned = re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()
    return re.sub(r"\s+", " ", cleaned)


def extract_first_hanzi(text: str) -> str:
    if not isinstance(text, str):
        return ""
    for ch in text:
        if re.match(r"[\u4e00-\u9fff]", ch):
            return ch
    return ""


@st.cache_data(show_spinner=False)
def fetch_stroke_svg_data(char: str):
    if not char:
        return {"ok": False, "error": "No character provided."}
    codepoint = ord(char)
    url = f"https://raw.githubusercontent.com/skishore/makemeahanzi/master/svgs/{codepoint}.svg"
    try:
        with urlopen(url, timeout=12) as resp:
            svg_text = resp.read().decode("utf-8")
    except HTTPError:
        return {"ok": False, "error": f"Stroke order not found for '{char}'."}
    except URLError:
        return {"ok": False, "error": "Network issue while loading stroke order."}
    except Exception:
        return {"ok": False, "error": "Could not load stroke order data."}

    ids = re.findall(r'id="make-me-a-hanzi-animation-(\d+)"', svg_text)
    stroke_numbers = sorted({int(i) + 1 for i in ids})
    stroke_count = len(stroke_numbers)

    return {
        "ok": True,
        "char": char,
        "codepoint": codepoint,
        "stroke_count": stroke_count,
        "sequence": stroke_numbers,
        "svg_text": svg_text,
        "source_url": url,
    }


def _normalize_expected_options(expected):
    if isinstance(expected, list):
        return [str(item).strip() for item in expected if str(item).strip()]
    text = str(expected).strip()
    return [text] if text else []


def _build_drag_drop_payload(sentences, word_bank):
    sentence_blocks = []
    expected_slots = []

    for sentence in sentences:
        text = str(sentence.get("text", ""))
        answers = sentence.get("answers", [])
        hanzi_hint = str(sentence.get("hanzi_hint", "")).strip()
        pinyin_hint = str(sentence.get("pinyin_hint", "")).strip()
        english_hint = str(sentence.get("english_hint", "")).strip()
        if not isinstance(answers, list):
            answers = [answers]

        segments = text.split("___")
        blank_count = max(len(segments) - 1, 0)
        if blank_count < len(answers):
            segments.extend([""] * (len(answers) - blank_count))
            blank_count = len(answers)
        elif blank_count > len(answers):
            answers = answers + [""] * (blank_count - len(answers))

        slot_indexes = []
        for answer in answers:
            expected_slots.append(_normalize_expected_options(answer))
            slot_indexes.append(len(expected_slots) - 1)

        sentence_blocks.append(
            {
                "segments": segments,
                "slot_indexes": slot_indexes,
                "hanzi_hint": hanzi_hint,
                "pinyin_hint": pinyin_hint,
                "english_hint": english_hint,
            }
        )

    tokens = [{"id": f"token_{i}", "text": str(word)} for i, word in enumerate(word_bank)]
    return {"sentences": sentence_blocks, "expected_slots": expected_slots, "tokens": tokens}


def _split_meaning_candidates(meaning_text: str):
    if not isinstance(meaning_text, str):
        return []
    raw_parts = re.split(r"[;/|；]+", meaning_text)
    parts = [p.strip() for p in raw_parts if p and p.strip()]
    # Keep short meaning units to make matching robust.
    dedup = []
    seen = set()
    for part in parts:
        key = normalize_en_answer(part)
        if key and key not in seen:
            seen.add(key)
            dedup.append(part)
    return dedup


_BLAND_EXAMPLE_PREFIXES = (
    "这是",
    "我在学习",
    "我有",
    "我在",
    "他在",
    "她在",
    "你在",
    "我吃了",
    "我是",
)


def _is_bland_example(example_zh: str, example_en: str = "") -> bool:
    zh = "" if pd.isna(example_zh) else str(example_zh).strip()
    en = "" if pd.isna(example_en) else str(example_en).strip()
    if not zh and not en:
        return True
    if zh and any(zh.startswith(prefix) for prefix in _BLAND_EXAMPLE_PREFIXES):
        return True
    if zh and "学习" in zh and ("“" in zh or "'" in zh):
        return True
    zh_core_len = len(re.sub(r"[^\u4e00-\u9fff]", "", zh))
    if zh and zh_core_len <= 4:
        return True
    return False


_LOW_QUALITY_EN_PATTERNS = (
    re.compile(r"^I to .+ every day\.$", re.IGNORECASE),
    re.compile(r"^This is a .+\.$", re.IGNORECASE),
    re.compile(r"^This is very .+\.$", re.IGNORECASE),
    re.compile(r"^He comes always\.$", re.IGNORECASE),
)

_LOW_QUALITY_ZH_PATTERNS = (
    re.compile(r"^我每天都.+[。！？]?$"),
    re.compile(r"^这是.+[。！？]?$"),
    re.compile(r"^这个非常.+[。！？]?$"),
    re.compile(r"^他总是.+[。！？]?$"),
)

_NO_PINYIN_CLOZE_PROMPTS = (
    "{context}  |  Missing Mandarin expression: ___",
    "Use this context to fill the Mandarin blank: {context}  |  ___",
    "Context sentence: {context}  |  Target Mandarin chunk: ___",
    "Choose the Mandarin expression that fits: {context}  |  ___",
)

_PROFESSIONAL_EXAMPLE_TEMPLATES = (
    (
        "在今天的讨论里，我们重点练习了“{hanzi}”这个表达。",
        "In today's discussion, we focused on the expression \"{hanzi}\" ({meaning}).",
    ),
    (
        "老师提醒我们，使用“{hanzi}”时要注意具体语境。",
        "Our teacher reminded us to pay attention to context when using \"{hanzi}\" ({meaning}).",
    ),
    (
        "这段对话里，“{hanzi}”是关键表达。",
        "In this dialogue, \"{hanzi}\" is a key expression meaning \"{meaning}.\"",
    ),
    (
        "为了让句子更自然，我把“{hanzi}”放在了核心位置。",
        "To make the sentence sound more natural, I placed \"{hanzi}\" ({meaning}) in a key part of the sentence.",
    ),
    (
        "在口语练习中，我们先理解“{hanzi}”，再完成整句表达。",
        "In speaking practice, we first understand \"{hanzi}\" ({meaning}) and then complete the full sentence.",
    ),
)

_CONVERSATION_PHRASE_OVERRIDES = {
    "两分钟": {"pinyin": "liǎng fēnzhōng", "english": "two minutes", "hsk": "HSK N/A", "frequency_rank": 400},
    "开完": {"pinyin": "kāiwán", "english": "to finish (an activity)", "hsk": "HSK N/A", "frequency_rank": 420},
    "截止日期": {"pinyin": "jiézhǐ rìqī", "english": "deadline", "hsk": "HSK N/A", "frequency_rank": 450},
    "周五": {"pinyin": "zhōuwǔ", "english": "Friday", "hsk": "HSK N/A", "frequency_rank": 380},
    "新版本": {"pinyin": "xīn bǎnběn", "english": "new version", "hsk": "HSK N/A", "frequency_rank": 430},
    "堵不堵": {"pinyin": "dǔ bu dǔ", "english": "whether traffic is congested", "hsk": "HSK N/A", "frequency_rank": 460},
    "高架": {"pinyin": "gāojià", "english": "elevated road", "hsk": "HSK N/A", "frequency_rank": 390},
    "有点": {"pinyin": "yǒudiǎn", "english": "a little; somewhat", "hsk": "HSK N/A", "frequency_rank": 260},
    "咖啡店": {"pinyin": "kāfēidiàn", "english": "coffee shop", "hsk": "HSK N/A", "frequency_rank": 410},
    "八点": {"pinyin": "bā diǎn", "english": "eight o'clock", "hsk": "HSK N/A", "frequency_rank": 300},
    "两点": {"pinyin": "liǎng diǎn", "english": "two o'clock", "hsk": "HSK N/A", "frequency_rank": 310},
    "二十": {"pinyin": "èrshí", "english": "twenty", "hsk": "HSK 1", "frequency_rank": 240},
    "走走": {"pinyin": "zǒuzou", "english": "to take a walk", "hsk": "HSK N/A", "frequency_rank": 470},
    "看展览": {"pinyin": "kàn zhǎnlǎn", "english": "to see an exhibition", "hsk": "HSK N/A", "frequency_rank": 440},
    "听起来": {"pinyin": "tīng qǐlái", "english": "to sound (as an impression)", "hsk": "HSK N/A", "frequency_rank": 320},
}


def _primary_meaning(meaning_text: str) -> str:
    candidates = _split_meaning_candidates(meaning_text)
    if not candidates:
        return "this expression"
    first = candidates[0].strip()
    first = re.sub(r"^\([^)]*\)\s*", "", first)
    return first.strip(" .,:;") or "this expression"


def _polish_english_sentence(text: str) -> str:
    cleaned = " ".join(str(text).replace("\u3000", " ").split()).strip()
    if not cleaned:
        return ""
    if cleaned[0].isalpha():
        cleaned = cleaned[0].upper() + cleaned[1:]
    if cleaned[-1] not in ".!?":
        cleaned += "."
    return cleaned


def _polish_chinese_sentence(text: str) -> str:
    cleaned = "".join(str(text).split()).strip()
    if not cleaned:
        return ""
    if cleaned[-1] not in "。！？":
        cleaned += "。"
    return cleaned


def _is_low_quality_english_example(example_en: str) -> bool:
    text = _polish_english_sentence(example_en)
    if not text:
        return True
    if re.search(r"[\u4e00-\u9fff]", text):
        return True
    if len(text.split()) <= 2:
        return True
    for pattern in _LOW_QUALITY_EN_PATTERNS:
        if pattern.match(text):
            return True
    return False


def _is_low_quality_chinese_example(example_zh: str, hanzi: str = "") -> bool:
    text = _polish_chinese_sentence(example_zh)
    if not text:
        return True
    if hanzi and hanzi not in text:
        return True
    if _is_bland_example(text):
        return True
    for pattern in _LOW_QUALITY_ZH_PATTERNS:
        if pattern.match(text):
            return True
    return False


def _generate_professional_example_pair(hanzi: str, meaning_text: str, variant_seed: int = 0):
    safe_hanzi = str(hanzi).strip() or "这个词"
    meaning = _primary_meaning(meaning_text)
    template = _PROFESSIONAL_EXAMPLE_TEMPLATES[variant_seed % len(_PROFESSIONAL_EXAMPLE_TEMPLATES)]
    zh = _polish_chinese_sentence(template[0].format(hanzi=safe_hanzi, meaning=meaning))
    en = _polish_english_sentence(template[1].format(hanzi=safe_hanzi, meaning=meaning))
    return zh, en


def _get_professional_learning_example(row, variant_seed: int = 0):
    hanzi = "" if pd.isna(row.get("hanzi_simplified", "")) else str(row.get("hanzi_simplified", "")).strip()
    meaning = "" if pd.isna(row.get("english_meanings", "")) else str(row.get("english_meanings", "")).strip()
    zh = "" if pd.isna(row.get("example_zh", "")) else str(row.get("example_zh", "")).strip()
    en = "" if pd.isna(row.get("example_en", "")) else str(row.get("example_en", "")).strip()

    zh_ok = not _is_low_quality_chinese_example(zh, hanzi=hanzi)
    en_ok = not _is_low_quality_english_example(en)

    if zh_ok and en_ok:
        return _polish_chinese_sentence(zh), _polish_english_sentence(en)

    return _generate_professional_example_pair(hanzi=hanzi, meaning_text=meaning, variant_seed=variant_seed)


def _prioritize_varied_examples(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    work = df.copy()
    zh = work["example_zh"].fillna("").astype(str)
    en = work["example_en"].fillna("").astype(str)
    zh_vals = [val.strip() for val in zh.tolist()]
    en_vals = [val.strip() for val in en.tolist()]
    bland_mask = pd.Series(
        [_is_bland_example(zh_val, en_val) for zh_val, en_val in zip(zh_vals, en_vals)],
        index=work.index,
    )
    zh_len = pd.Series(
        [len(re.sub(r"[^\u4e00-\u9fff]", "", zh_val)) for zh_val in zh_vals],
        index=work.index,
    )
    punctuation_bonus = pd.Series(
        [1 if any(ch in "，。！？?!" for ch in zh_val) else 0 for zh_val in zh_vals],
        index=work.index,
    )
    en_len = pd.Series([len(en_val) for en_val in en_vals], index=work.index)
    work["_example_score"] = (
        (~bland_mask).astype(int) * 4
        + (zh_len >= 8).astype(int)
        + punctuation_bonus
        + (en_len >= 16).astype(int)
    )
    return work.sort_values(by=["_example_score", "frequency_rank"], ascending=[False, True]).drop(columns=["_example_score"])


def _extract_unique_hanzi_chars(text: str):
    if not isinstance(text, str):
        return []
    chars = []
    seen = set()
    for ch in text:
        if re.match(r"[\u4e00-\u9fff]", ch) and ch not in seen:
            seen.add(ch)
            chars.append(ch)
    return chars


def _build_expression_lookup(df: pd.DataFrame):
    cols = ["hanzi_simplified", "pinyin", "english_meanings", "character_count", "official_hsk", "frequency_rank"]
    work = df[cols].copy()
    work["hanzi_simplified"] = work["hanzi_simplified"].fillna("").astype(str).str.strip()
    work["pinyin"] = work["pinyin"].fillna("").astype(str).str.strip()
    work["english_meanings"] = work["english_meanings"].fillna("").astype(str).str.strip()
    work["character_count"] = pd.to_numeric(work["character_count"], errors="coerce").fillna(0).astype(int)
    work["frequency_rank"] = pd.to_numeric(work["frequency_rank"], errors="coerce").fillna(999999)
    work = work[work["hanzi_simplified"] != ""]
    work = work.sort_values(by=["character_count", "frequency_rank"], ascending=[False, True])
    work = work.drop_duplicates(subset=["hanzi_simplified"], keep="first")

    lookup = {}
    for _, row in work.iterrows():
        hanzi = str(row["hanzi_simplified"]).strip()
        hsk_val = row["official_hsk"]
        hsk_text = f"HSK {int(hsk_val)}" if pd.notna(hsk_val) else "HSK N/A"
        lookup[hanzi] = {
            "hanzi": hanzi,
            "pinyin": str(row["pinyin"]).strip(),
            "english": str(row["english_meanings"]).strip(),
            "length": int(row["character_count"]),
            "hsk": hsk_text,
            "frequency_rank": int(row["frequency_rank"]),
        }

    for phrase, info in _CONVERSATION_PHRASE_OVERRIDES.items():
        key = str(phrase).strip()
        if not key:
            continue
        if key not in lookup:
            lookup[key] = {
                "hanzi": key,
                "pinyin": str(info.get("pinyin", "")).strip(),
                "english": str(info.get("english", "")).strip(),
                "length": len(key),
                "hsk": str(info.get("hsk", "HSK N/A")).strip() or "HSK N/A",
                "frequency_rank": int(info.get("frequency_rank", 500)),
            }
    return lookup


def _extract_line_learning_chunks(line_text: str, expression_lookup: dict, max_char_len: int = 7, max_items: int = 14):
    if not isinstance(line_text, str) or not line_text.strip():
        return []
    hanzi_only = "".join(re.findall(r"[\u4e00-\u9fff]", line_text))
    if not hanzi_only:
        return []

    phrase_set = set(expression_lookup.keys())
    ordered = []
    seen = set()

    def add_chunk(chunk: str):
        if chunk in seen:
            return False
        info = expression_lookup.get(chunk)
        if not info:
            return False
        seen.add(chunk)
        ordered.append(info)
        return True

    i = 0
    text_len = len(hanzi_only)
    while i < text_len:
        matched = None
        max_len = min(max_char_len, text_len - i)
        for size in range(max_len, 0, -1):
            chunk = hanzi_only[i : i + size]
            if chunk in phrase_set:
                matched = chunk
                break
        if matched:
            add_chunk(matched)
            i += len(matched)
        else:
            i += 1

    extras = []
    for start in range(text_len):
        max_len = min(max_char_len, text_len - start)
        for size in range(max_len, 1, -1):
            chunk = hanzi_only[start : start + size]
            if chunk in phrase_set and chunk not in seen:
                info = expression_lookup.get(chunk)
                if info:
                    extras.append(info)
    extras = sorted(extras, key=lambda x: (-x["length"], x["frequency_rank"]))
    for info in extras:
        if len(ordered) >= max_items:
            break
        add_chunk(info["hanzi"])

    for ch in _extract_unique_hanzi_chars(line_text):
        if len(ordered) >= max_items:
            break
        add_chunk(ch)

    return ordered[:max_items]


def _is_hanzi_char(ch: str) -> bool:
    return bool(re.match(r"[\u4e00-\u9fff]", ch or ""))


def _segment_hanzi_run_for_click_selection(run_text: str, expression_lookup: dict, max_char_len: int = 4, min_click_len: int = 2):
    run = "" if not isinstance(run_text, str) else run_text
    if not run:
        return []

    n = len(run)
    best_score = [-10**9] * (n + 1)
    choice = [None] * (n + 1)
    best_score[n] = 0.0
    choice[n] = ("end", "", 0)

    for i in range(n - 1, -1, -1):
        # Fallback: keep this character as plain text (not clickable).
        plain_score = -1.8 + best_score[i + 1]
        best_score[i] = plain_score
        choice[i] = ("plain", run[i], 1)

        max_size = min(max_char_len, n - i)
        for size in range(min_click_len, max_size + 1):
            candidate = run[i : i + size]
            info = expression_lookup.get(candidate)
            if not info:
                continue

            freq = float(info.get("frequency_rank", 999999))
            length_bonus = size * 3.0
            two_char_bonus = 1.2 if size == 2 else 0.0
            freq_bonus = max(0.0, 2.0 - min(freq, 30000.0) / 15000.0)
            score = length_bonus + two_char_bonus + freq_bonus + best_score[i + size]

            if score > best_score[i]:
                best_score[i] = score
                choice[i] = ("click", candidate, size)

    segments = []
    i = 0
    while i < n:
        kind, text, size = choice[i] if choice[i] else ("plain", run[i], 1)
        segments.append({"text": text, "clickable": kind == "click"})
        i += max(size, 1)
    return segments


def _build_sentence_click_tokens(line_text: str, expression_lookup: dict, max_char_len: int = 4):
    text = "" if not isinstance(line_text, str) else line_text
    if not text:
        return []

    tokens = []
    i = 0
    while i < len(text):
        ch = text[i]
        if not _is_hanzi_char(ch):
            tokens.append({"text": ch, "clickable": False})
            i += 1
            continue

        j = i
        while j < len(text) and _is_hanzi_char(text[j]):
            j += 1
        run = text[i:j]
        run_segments = _segment_hanzi_run_for_click_selection(
            run_text=run,
            expression_lookup=expression_lookup,
            max_char_len=max_char_len,
            min_click_len=2,
        )
        for segment in run_segments:
            seg_text = str(segment.get("text", ""))
            if not seg_text:
                continue

            if bool(segment.get("clickable")) and seg_text in expression_lookup:
                info = expression_lookup.get(seg_text, {})
                primary = _primary_meaning(str(info.get("english", "")))
                tokens.append(
                    {
                        "text": seg_text,
                        "clickable": True,
                        "pinyin": str(info.get("pinyin", "")).strip(),
                        "meaning_primary": primary,
                        "meaning_full": str(info.get("english", "")).strip(),
                        "hsk": str(info.get("hsk", "HSK N/A")).strip() or "HSK N/A",
                    }
                )
            else:
                tokens.append({"text": seg_text, "clickable": False})
        i = j

    return tokens


def render_conversation_line_selector(activity_id: str, role: str, hanzi: str, pinyin: str, english: str, expression_lookup: dict):
    tokens = _build_sentence_click_tokens(hanzi, expression_lookup)
    if not tokens:
        st.markdown(f"**{role}** {hanzi}")
        if pinyin:
            st.caption(pinyin)
        if english:
            st.caption(english)
        return

    if not any(bool(t.get("clickable")) for t in tokens):
        st.markdown(f"**{role}** {hanzi}")
        if pinyin:
            st.caption(pinyin)
        if english:
            st.caption(english)
        return

    safe_id = re.sub(r"[^a-zA-Z0-9_]+", "_", activity_id)
    payload = json.dumps(
        {
            "role": role,
            "hanzi": hanzi,
            "pinyin": pinyin,
            "english": english,
            "tokens": tokens,
            "palette": ["#2563eb", "#0ea5e9", "#16a34a", "#f59e0b", "#14b8a6"],
        },
        ensure_ascii=False,
    )

    html = f"""
    <div class="dcx-wrap" id="dcx-wrap-{safe_id}">
      <div class="dcx-selection">
        <div class="dcx-selection-main">
          <div id="dcx-selected-pinyin-{safe_id}" class="dcx-selected-pinyin">-</div>
          <div id="dcx-selected-hanzi-{safe_id}" class="dcx-selected-hanzi">-</div>
          <div id="dcx-selected-meaning-{safe_id}" class="dcx-selected-meaning">-</div>
          <div id="dcx-selected-full-{safe_id}" class="dcx-selected-full"></div>
        </div>
        <div id="dcx-selected-hsk-{safe_id}" class="dcx-selected-hsk">HSK N/A</div>
      </div>
      <div class="dcx-zh-block">
        <div class="dcx-speaker">{role}</div>
        <div id="dcx-sentence-{safe_id}" class="dcx-sentence"></div>
      </div>
      <div class="dcx-en">{english}</div>
    </div>
    <style>
      .dcx-wrap {{
        border: 1px solid #dbe2ea;
        border-radius: 14px;
        background: #ffffff;
        padding: 12px;
        margin-bottom: 10px;
      }}
      .dcx-selection {{
        display: flex;
        align-items: flex-start;
        justify-content: space-between;
        gap: 10px;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        background: #f8fafc;
        padding: 12px;
      }}
      .dcx-selection-main {{
        min-width: 0;
      }}
      .dcx-selected-pinyin {{
        font-size: 30px;
        line-height: 1.2;
        color: #111827;
      }}
      .dcx-selected-hanzi {{
        font-size: 48px;
        line-height: 1.1;
        font-weight: 700;
        color: #0f172a;
      }}
      .dcx-selected-meaning {{
        margin-top: 2px;
        font-size: 22px;
        color: #111827;
      }}
      .dcx-selected-full {{
        margin-top: 6px;
        font-size: 14px;
        color: #475569;
      }}
      .dcx-selected-hsk {{
        border: 1px solid #f59e0b;
        color: #b45309;
        border-radius: 999px;
        padding: 4px 10px;
        font-size: 14px;
        font-weight: 700;
        background: #fffbeb;
        white-space: nowrap;
      }}
      .dcx-zh-block {{
        margin-top: 10px;
        border-radius: 12px;
        background: #ffffff;
      }}
      .dcx-speaker {{
        font-size: 16px;
        font-weight: 700;
        color: #334155;
        margin-bottom: 6px;
      }}
      .dcx-sentence {{
        display: flex;
        flex-wrap: wrap;
        align-items: flex-end;
        gap: 3px 5px;
      }}
      .dcx-token {{
        border: none;
        background: transparent;
        padding: 0 1px;
        cursor: pointer;
        display: inline-flex;
        flex-direction: column;
        align-items: center;
        justify-content: flex-end;
        border-bottom: 2px solid transparent;
        min-height: 72px;
      }}
      .dcx-token-pinyin {{
        font-size: 14px;
        line-height: 1.1;
        color: #475569;
        margin-bottom: 2px;
      }}
      .dcx-token-hanzi {{
        font-size: 42px;
        line-height: 1.1;
        font-weight: 700;
        color: #0f172a;
      }}
      .dcx-token.active {{
        border-bottom-color: var(--token-color, #2563eb);
      }}
      .dcx-plain {{
        font-size: 42px;
        line-height: 1.1;
        color: #111827;
        padding-bottom: 2px;
      }}
      .dcx-en {{
        margin-top: 10px;
        border-radius: 10px;
        background: #f3f4f6;
        color: #111827;
        font-size: 20px;
        line-height: 1.45;
        padding: 10px 12px;
      }}
      @media (max-width: 760px) {{
        .dcx-selected-pinyin {{ font-size: 24px; }}
        .dcx-selected-hanzi {{ font-size: 40px; }}
        .dcx-selected-meaning {{ font-size: 18px; }}
        .dcx-token-hanzi {{ font-size: 34px; }}
        .dcx-plain {{ font-size: 34px; }}
        .dcx-en {{ font-size: 17px; }}
      }}
    </style>
    <script>
      (() => {{
        const payload = {payload};
        const tokens = payload.tokens || [];
        const palette = payload.palette || ["#2563eb"];
        const pinyinEl = document.getElementById("dcx-selected-pinyin-{safe_id}");
        const hanziEl = document.getElementById("dcx-selected-hanzi-{safe_id}");
        const meaningEl = document.getElementById("dcx-selected-meaning-{safe_id}");
        const fullEl = document.getElementById("dcx-selected-full-{safe_id}");
        const hskEl = document.getElementById("dcx-selected-hsk-{safe_id}");
        const sentenceEl = document.getElementById("dcx-sentence-{safe_id}");

        let activeIdx = tokens.findIndex((t) => t && t.clickable);
        if (activeIdx < 0) activeIdx = 0;

        const esc = (val) => {{
          const d = document.createElement("div");
          d.textContent = val || "";
          return d.innerHTML;
        }};

        const renderSelection = () => {{
          const token = tokens[activeIdx] || {{}};
          pinyinEl.textContent = token.pinyin || payload.pinyin || "-";
          hanziEl.textContent = token.text || payload.hanzi || "-";
          meaningEl.textContent = token.meaning_primary || payload.english || "-";
          fullEl.textContent =
            token.meaning_full && token.meaning_full !== token.meaning_primary ? token.meaning_full : "";
          hskEl.textContent = token.hsk || "HSK N/A";
        }};

        const renderTokens = () => {{
          sentenceEl.innerHTML = "";
          let clickableOrder = 0;
          tokens.forEach((token, idx) => {{
            if (token.clickable) {{
              const btn = document.createElement("button");
              btn.type = "button";
              btn.className = "dcx-token" + (idx === activeIdx ? " active" : "");
              btn.style.setProperty("--token-color", palette[clickableOrder % palette.length]);
              clickableOrder += 1;
              btn.innerHTML = `
                <span class="dcx-token-pinyin">${{esc(token.pinyin || "")}}</span>
                <span class="dcx-token-hanzi">${{esc(token.text || "")}}</span>
              `;
              btn.addEventListener("click", () => {{
                activeIdx = idx;
                renderTokens();
                renderSelection();
              }});
              sentenceEl.appendChild(btn);
            }} else {{
              const span = document.createElement("span");
              span.className = "dcx-plain";
              span.textContent = token.text || "";
              sentenceEl.appendChild(span);
            }}
          }});
        }};

        renderTokens();
        renderSelection();
      }})();
    </script>
    """

    components.html(html, height=380, scrolling=False)


def _is_english_semantic_match(user_answer: str, meaning_text: str, example_en: str = ""):
    user_norm = normalize_en_answer(user_answer)
    if not user_norm:
        return False

    candidates = _split_meaning_candidates(meaning_text)
    if isinstance(example_en, str) and example_en.strip():
        candidates.append(example_en.strip())

    for cand in candidates:
        cand_norm = normalize_en_answer(cand)
        if not cand_norm:
            continue

        if user_norm == cand_norm:
            return True
        if len(user_norm) >= 4 and (user_norm in cand_norm or cand_norm in user_norm):
            return True

        ratio = SequenceMatcher(a=user_norm, b=cand_norm).ratio()
        if ratio >= 0.74:
            return True

        user_tokens = set(user_norm.split())
        cand_tokens = set(cand_norm.split())
        if cand_tokens:
            overlap = user_tokens.intersection(cand_tokens)
            if len(overlap) >= 1 and (len(overlap) / max(len(cand_tokens), 1)) >= 0.5:
                return True
    return False


def _prepare_quiz_source_df(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    for col in ["hanzi_simplified", "pinyin", "english_meanings", "example_zh", "example_en"]:
        if col not in work.columns:
            work[col] = ""
        work[col] = work[col].fillna("").astype(str).str.strip()
    return work


def _apply_quiz_prompt_filter(df: pd.DataFrame, question_count: int, prompt_style: str, quiz_kind: str) -> pd.DataFrame:
    use_hanzi_pinyin_prompt = prompt_style == "hanzi_pinyin"
    work = _prepare_quiz_source_df(df)
    work = work[work["hanzi_simplified"] != ""]

    if quiz_kind == "translation":
        work = work[work["english_meanings"] != ""]

    if use_hanzi_pinyin_prompt:
        work = work[work["pinyin"] != ""]
        zh_quality = work.apply(
            lambda r: not _is_low_quality_chinese_example(r["example_zh"], hanzi=r["hanzi_simplified"]),
            axis=1,
        )
        strict = work[zh_quality]
    else:
        base_mask = (work["english_meanings"] != "") | (work["example_en"] != "")
        work = work[base_mask]
        en_quality = work["example_en"].apply(lambda x: not _is_low_quality_english_example(x))
        strict = work[en_quality]

    min_floor = min(len(work), max(8, int(question_count * 0.65)))
    if len(strict) >= min_floor:
        return _prioritize_varied_examples(strict)
    return _prioritize_varied_examples(work)


def _build_sentence_pinyin_display(sentence_text: str, expression_lookup: dict, max_char_len: int = 4) -> str:
    text = "" if not isinstance(sentence_text, str) else sentence_text
    if not text:
        return ""

    punctuation = set("，。！？、,.!?;:：；")
    output_tokens = []
    i = 0

    while i < len(text):
        if text.startswith("___", i):
            output_tokens.append("___")
            i += 3
            continue

        ch = text[i]
        if _is_hanzi_char(ch):
            j = i
            while j < len(text) and _is_hanzi_char(text[j]):
                j += 1
            run = text[i:j]
            k = 0
            while k < len(run):
                matched = None
                max_size = min(max_char_len, len(run) - k)
                for size in range(max_size, 0, -1):
                    candidate = run[k : k + size]
                    info = expression_lookup.get(candidate)
                    if info and str(info.get("pinyin", "")).strip():
                        matched = (candidate, str(info.get("pinyin", "")).strip())
                        break
                if matched:
                    output_tokens.append(matched[1])
                    k += len(matched[0])
                else:
                    output_tokens.append(run[k])
                    k += 1
            i = j
            continue

        if ch.isspace():
            output_tokens.append(" ")
        else:
            output_tokens.append(ch)
        i += 1

    built = ""
    for token in output_tokens:
        if token == " ":
            if built and not built.endswith(" "):
                built += " "
            continue

        if token in punctuation:
            built = built.rstrip() + token + " "
            continue

        if built and not built.endswith(" "):
            built += " "
        built += token

    return re.sub(r"\s+", " ", built).strip()


def _paginate_quiz_sentences(sentences, word_bank, page_size: int = 8):
    if not sentences:
        return []

    bank = [str(w).strip() for w in word_bank if str(w).strip()]
    pages = []
    for start in range(0, len(sentences), page_size):
        page_sentences = sentences[start : start + page_size]
        answers = []
        for sent in page_sentences:
            raw_answers = sent.get("answers", [])
            if not isinstance(raw_answers, list):
                raw_answers = [raw_answers]
            for raw in raw_answers:
                opts = _normalize_expected_options(raw)
                if opts:
                    answers.append(opts[0])

        page_answers = list(dict.fromkeys([a for a in answers if a]))
        distractor_pool = [w for w in bank if w not in set(page_answers)]
        distractor_count = min(len(distractor_pool), max(5, int(len(page_answers) * 0.8)))
        page_word_bank = list(dict.fromkeys(page_answers + distractor_pool[:distractor_count]))
        pages.append({"sentences": page_sentences, "word_bank": page_word_bank})
    return pages


def render_paginated_drag_drop_activity(activity_id, title, prompt, sentences, word_bank, page_size=8):
    pages = _paginate_quiz_sentences(sentences=sentences, word_bank=word_bank, page_size=page_size)
    if not pages:
        st.warning("No quiz items available for this activity.")
        return

    if len(pages) == 1:
        render_drag_drop_activity(
            activity_id=activity_id,
            title=title,
            prompt=prompt,
            sentences=pages[0]["sentences"],
            word_bank=pages[0]["word_bank"],
            height=min(1700, 360 + int(len(pages[0]["sentences"]) * 86)),
        )
        return

    st.caption(f"Questions are split into {len(pages)} pages, each with its own smaller word bank.")
    page_idx = st.selectbox(
        "Quiz page",
        options=list(range(len(pages))),
        format_func=lambda idx: f"Page {idx + 1} ({len(pages[idx]['sentences'])} questions)",
        key=f"{activity_id}_page_pick",
    )
    page = pages[int(page_idx)]
    render_drag_drop_activity(
        activity_id=f"{activity_id}_p{int(page_idx) + 1}",
        title=f"{title} - Page {int(page_idx) + 1}/{len(pages)}",
        prompt=prompt,
        sentences=page["sentences"],
        word_bank=page["word_bank"],
        height=min(1700, 360 + int(len(page["sentences"]) * 86)),
    )


def _prepare_cloze_quiz_items(df: pd.DataFrame, question_count: int, prompt_style: str):
    use_hanzi_pinyin_prompt = prompt_style == "hanzi_pinyin"
    # Cloze sentence content is shared across both modes.
    # "No pinyin" should show the same Hanzi question, only without pinyin support text.
    work = _apply_quiz_prompt_filter(
        df=df,
        question_count=question_count,
        prompt_style="hanzi_pinyin",
        quiz_kind="cloze",
    )

    if work.empty:
        return [], []

    expression_lookup_source = vocab if "vocab" in globals() else work
    expression_lookup = _build_expression_lookup(expression_lookup_source)

    sample_size = min(question_count, len(work))
    top_pool_size = min(len(work), max(sample_size * 4, sample_size))
    sampled = work.head(top_pool_size).sample(sample_size, random_state=42).reset_index(drop=True)

    sentences = []
    answers = []
    for idx, row in sampled.iterrows():
        answer = str(row["hanzi_simplified"]).strip()
        pinyin = str(row["pinyin"]).strip()
        english = str(row["english_meanings"]).strip()
        ex_zh, ex_en = _get_professional_learning_example(row, variant_seed=idx + 11)
        en_base = ex_en or _polish_english_sentence(_primary_meaning(english))

        base = ex_zh
        if answer and answer in base:
            text = base.replace(answer, "___", 1)
        else:
            text = f"{base}  （填空：___）"
        hanzi_hint = ""
        pinyin_hint = (
            _build_sentence_pinyin_display(text, expression_lookup) or (f"Target pinyin: {pinyin}" if pinyin else "")
            if use_hanzi_pinyin_prompt
            else ""
        )
        english_hint = f"English context: {en_base}" if en_base else f"Meaning: {_primary_meaning(english)}"

        sentences.append(
            {
                "text": text,
                "answers": [answer],
                "hanzi_hint": hanzi_hint,
                "pinyin_hint": pinyin_hint,
                "english_hint": english_hint,
            }
        )
        answers.append(answer)

    # Add distractors so the bank is larger than question count.
    extra_pool = work[~work["hanzi_simplified"].isin(set(answers))]["hanzi_simplified"].dropna().astype(str)
    extra_pool = [x.strip() for x in extra_pool.tolist() if x and x.strip()]
    extra_pool = list(dict.fromkeys(extra_pool))
    extra_count = min(max(6, int(question_count * 0.35)), len(extra_pool))
    distractors = extra_pool[:extra_count]

    word_bank = list(dict.fromkeys(answers + distractors))
    return sentences, word_bank


def render_translation_quiz(activity_id: str, df: pd.DataFrame, question_count: int, prompt_style: str):
    use_hanzi_pinyin_prompt = prompt_style == "hanzi_pinyin"
    st.markdown("#### English Translation Quiz")
    st.caption("Type the English meaning. Grading is flexible: close meaning is accepted.")

    work = _apply_quiz_prompt_filter(
        df=df,
        question_count=question_count,
        prompt_style=prompt_style,
        quiz_kind="translation",
    )

    if work.empty:
        st.warning("Not enough vocabulary rows for translation quiz with current filters.")
        return

    sample_size = min(question_count, len(work))
    top_pool_size = min(len(work), max(sample_size * 4, sample_size))
    sampled = work.head(top_pool_size).sample(sample_size, random_state=99).reset_index(drop=True)
    expression_lookup_source = vocab if "vocab" in globals() else work
    expression_lookup = _build_expression_lookup(expression_lookup_source)

    with st.form(f"{activity_id}_translation_form"):
        user_answers = []
        example_pairs = []
        for i, row in sampled.iterrows():
            hanzi = str(row["hanzi_simplified"]).strip()
            pinyin = str(row["pinyin"]).strip()
            zh_ex, en_ex = _get_professional_learning_example(row, variant_seed=i + 101)
            example_pairs.append((zh_ex, en_ex))
            pinyin_line = pinyin if pinyin else "(no pinyin available)"

            if use_hanzi_pinyin_prompt:
                sentence_zh = zh_ex if zh_ex else f"请用“{hanzi}”完成句子。"
                sentence_pinyin = _build_sentence_pinyin_display(sentence_zh, expression_lookup) or pinyin_line
                st.markdown(f"**{i + 1}. Translate (full sentence):**")
                st.markdown(f"{sentence_zh}  \n{sentence_pinyin}")
                st.caption(f"Target word: {hanzi} ({pinyin_line})")
            else:
                st.markdown(f"**{i + 1}. Translate (no pinyin):** `{hanzi}`")
                st.caption(f"Hanzi hint: {hanzi}")
                if en_ex:
                    st.caption(f"English context: {en_ex}")

            user_answers.append(
                st.text_input(
                    f"Answer {i + 1}",
                    key=f"{activity_id}_ans_{i}",
                    placeholder="Type English meaning",
                )
            )

        submitted = st.form_submit_button("Check Translation Answers")

    if submitted:
        score = 0
        st.markdown("**Result**")
        for i, row in sampled.iterrows():
            expected = str(row["english_meanings"]).strip()
            en_ex = example_pairs[i][1] if i < len(example_pairs) else ""
            user_val = user_answers[i]
            ok = _is_english_semantic_match(user_val, expected, en_ex)
            if ok:
                score += 1
                st.success(f"Q{i + 1}: Accepted")
            else:
                st.error(f"Q{i + 1}: Not matched yet")
            st.caption(f"Your answer: {user_val if user_val.strip() else '(empty)'}")
            st.caption(f"Accepted meanings: {expected}")

        total = len(sampled)
        st.markdown(f"**Score: {score}/{total}**")
        st.progress(score / total if total else 0)
        if score == total and total > 0:
            st.balloons()


def render_cue_card_deck(activity_id: str, cue_df: pd.DataFrame):
    if "vocab" in globals() and isinstance(vocab, pd.DataFrame) and not vocab.empty:
        expression_lookup = _build_expression_lookup(vocab)
    else:
        expression_lookup = {}
        for _, row in cue_df.iterrows():
            hz = "" if pd.isna(row.get("hanzi_simplified", "")) else str(row.get("hanzi_simplified", "")).strip()
            py = "" if pd.isna(row.get("pinyin", "")) else str(row.get("pinyin", "")).strip()
            en = "" if pd.isna(row.get("english_meanings", "")) else str(row.get("english_meanings", "")).strip()
            if hz:
                expression_lookup[hz] = {
                    "hanzi": hz,
                    "pinyin": py,
                    "english": en,
                    "length": len(hz),
                    "hsk": "HSK N/A",
                    "frequency_rank": 999999,
                }

    data = []
    for idx, row in cue_df.iterrows():
        hanzi = "" if pd.isna(row["hanzi_simplified"]) else str(row["hanzi_simplified"]).strip()
        pinyin = "" if pd.isna(row["pinyin"]) else str(row["pinyin"]).strip()
        meaning = "" if pd.isna(row["english_meanings"]) else str(row["english_meanings"]).strip()
        ex_zh, ex_en = _get_professional_learning_example(row, variant_seed=int(idx) + 205)
        ex_py = _build_sentence_pinyin_display(ex_zh, expression_lookup) if ex_zh else ""
        data.append(
            {
                "hanzi": hanzi,
                "pinyin": pinyin,
                "meaning": meaning,
                "ex_zh": ex_zh,
                "ex_py": ex_py,
                "ex_en": ex_en,
            }
        )

    if not data:
        st.warning("No cue cards available for current filters.")
        return

    safe_id = re.sub(r"[^a-zA-Z0-9_]+", "_", activity_id)
    payload = json.dumps(data, ensure_ascii=False)
    html = f"""
    <div class="cue-wrap" id="cue-wrap-{safe_id}">
      <div class="cue-help">Tap anywhere on the card to flip front/back. Use arrows inside card for previous/next.</div>
      <div id="cue-index-{safe_id}" class="cue-index"></div>
      <div id="cue-card-{safe_id}" class="cue-card" role="button" tabindex="0">
        <button id="cue-prev-{safe_id}" class="cue-arrow cue-arrow-left" aria-label="Previous">◀</button>
        <button id="cue-next-{safe_id}" class="cue-arrow cue-arrow-right" aria-label="Next">▶</button>
        <div id="cue-face-{safe_id}" class="cue-face"></div>
      </div>
      <div class="cue-small-actions">
        <button id="cue-random-{safe_id}" type="button">Random</button>
      </div>
    </div>
    <style>
      .cue-wrap {{
        width: 100%;
      }}
      .cue-help {{
        font-size: 13px;
        color: #9ca3af;
        margin-bottom: 8px;
      }}
      .cue-index {{
        font-size: 13px;
        color: #9ca3af;
        margin-bottom: 8px;
      }}
      .cue-card {{
        position: relative;
        border: 1px solid #dbe2ea;
        border-radius: 14px;
        background: #f8fafc;
        min-height: 280px;
        padding: 28px 52px;
        cursor: pointer;
        user-select: none;
      }}
      .cue-face-front-hanzi {{
        font-size: 56px;
        line-height: 1.1;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 10px;
      }}
      .cue-face-front-pinyin {{
        font-size: 24px;
        font-weight: 600;
        color: #1f2937;
        margin-bottom: 10px;
      }}
      .cue-face-front-meaning {{
        font-size: 18px;
        color: #111827;
      }}
      .cue-back-label {{
        font-size: 14px;
        font-weight: 700;
        color: #334155;
        margin-bottom: 6px;
      }}
      .cue-back-zh {{
        font-size: 28px;
        line-height: 1.45;
        color: #111827;
        margin-bottom: 8px;
      }}
      .cue-back-pinyin {{
        font-size: 20px;
        line-height: 1.5;
        color: #334155;
        margin-bottom: 14px;
      }}
      .cue-back-en {{
        font-size: 18px;
        line-height: 1.5;
        color: #111827;
      }}
      .cue-arrow {{
        position: absolute;
        top: 50%;
        transform: translateY(-50%);
        width: 34px;
        height: 34px;
        border: 1px solid #94a3b8;
        border-radius: 999px;
        background: rgba(255, 255, 255, 0.93);
        color: #111827;
        font-weight: 700;
        cursor: pointer;
      }}
      .cue-arrow-left {{
        left: 12px;
      }}
      .cue-arrow-right {{
        right: 12px;
      }}
      .cue-small-actions {{
        margin-top: 8px;
        display: flex;
        justify-content: flex-end;
      }}
      .cue-small-actions button {{
        border: 1px solid #cbd5e1;
        border-radius: 8px;
        background: #ffffff;
        color: #111827;
        padding: 5px 10px;
        cursor: pointer;
      }}
      @media (max-width: 680px) {{
        .cue-card {{
          min-height: 250px;
          padding: 20px 44px;
        }}
        .cue-face-front-hanzi {{
          font-size: 44px;
        }}
        .cue-face-front-pinyin {{
          font-size: 20px;
        }}
        .cue-back-zh {{
          font-size: 23px;
        }}
        .cue-back-pinyin {{
          font-size: 17px;
        }}
        .cue-back-en {{
          font-size: 17px;
        }}
      }}
    </style>
    <script>
      (() => {{
        const cards = {payload};
        const indexEl = document.getElementById("cue-index-{safe_id}");
        const cardEl = document.getElementById("cue-card-{safe_id}");
        const faceEl = document.getElementById("cue-face-{safe_id}");
        const prevBtn = document.getElementById("cue-prev-{safe_id}");
        const nextBtn = document.getElementById("cue-next-{safe_id}");
        const randomBtn = document.getElementById("cue-random-{safe_id}");

        let idx = 0;
        let showBack = false;

        const esc = (val) => {{
          const d = document.createElement("div");
          d.textContent = val || "";
          return d.innerHTML;
        }};

        const render = () => {{
          const c = cards[idx] || {{}};
          const hanzi = c.hanzi ? esc(c.hanzi) : "—";
          const pinyin = c.pinyin ? esc(c.pinyin) : "—";
          const meaning = c.meaning ? esc(c.meaning) : "—";
          const exZh = c.ex_zh ? esc(c.ex_zh) : "No sentence available.";
          const exPy = c.ex_py ? esc(c.ex_py) : "No pinyin available.";
          const exEn = c.ex_en ? esc(c.ex_en) : "No English translation available.";

          indexEl.textContent = "Card " + (idx + 1) + "/" + cards.length;
          if (!showBack) {{
            faceEl.innerHTML = `
              <div class="cue-face-front-hanzi">${{hanzi}}</div>
              <div class="cue-face-front-pinyin">${{pinyin}}</div>
              <div class="cue-face-front-meaning">${{meaning}}</div>
            `;
          }} else {{
            faceEl.innerHTML = `
              <div class="cue-back-label">How to use it in a sentence</div>
              <div class="cue-back-zh">${{exZh}}</div>
              <div class="cue-back-label">Pinyin</div>
              <div class="cue-back-pinyin">${{exPy}}</div>
              <div class="cue-back-label">English translation</div>
              <div class="cue-back-en">${{exEn}}</div>
            `;
          }}
        }};

        const move = (step) => {{
          idx = (idx + step + cards.length) % cards.length;
          showBack = false;
          render();
        }};

        cardEl.addEventListener("click", (evt) => {{
          if (evt.target === prevBtn || evt.target === nextBtn || evt.target === randomBtn) {{
            return;
          }}
          showBack = !showBack;
          render();
        }});

        cardEl.addEventListener("keydown", (evt) => {{
          if (evt.key === "Enter" || evt.key === " ") {{
            evt.preventDefault();
            showBack = !showBack;
            render();
          }} else if (evt.key === "ArrowLeft") {{
            evt.preventDefault();
            move(-1);
          }} else if (evt.key === "ArrowRight") {{
            evt.preventDefault();
            move(1);
          }}
        }});

        prevBtn.addEventListener("click", (evt) => {{
          evt.stopPropagation();
          move(-1);
        }});

        nextBtn.addEventListener("click", (evt) => {{
          evt.stopPropagation();
          move(1);
        }});

        randomBtn.addEventListener("click", (evt) => {{
          evt.stopPropagation();
          idx = Math.floor(Math.random() * cards.length);
          showBack = false;
          render();
        }});

        render();
      }})();
    </script>
    """
    components.html(html, height=470, scrolling=False)


def render_drag_drop_activity(activity_id, title, prompt, sentences, word_bank, height=None):
    st.markdown(f"#### {title}")
    st.caption(prompt)

    payload = _build_drag_drop_payload(sentences=sentences, word_bank=word_bank)
    if not payload["expected_slots"]:
        st.warning("No blanks configured for this activity.")
        return

    payload_json = json.dumps(payload, ensure_ascii=False)
    safe_id = re.sub(r"[^a-zA-Z0-9_]+", "_", activity_id)

    html = f"""
    <div id="dd-wrap-{safe_id}" class="dd-wrap">
      <div class="dd-toolbar">
        <button id="check-btn-{safe_id}" type="button">Check Answers</button>
        <button id="reset-btn-{safe_id}" type="button">Reset</button>
        <span class="dd-tip">Drag words into each blank. Click a filled blank to return it to the word bank.</span>
      </div>
      <div id="dd-sentences-{safe_id}" class="dd-sentences"></div>
      <div class="dd-bank-label">Word Bank</div>
      <div id="dd-bank-{safe_id}" class="dd-bank"></div>
      <div id="dd-result-{safe_id}" class="dd-result"></div>
    </div>
    <style>
      .dd-wrap {{
        border: 1px solid #dbe2ea;
        border-radius: 12px;
        padding: 12px 12px 10px 12px;
        background: #f9fbfd;
      }}
      .dd-toolbar {{
        display: flex;
        gap: 8px;
        align-items: center;
        flex-wrap: wrap;
      }}
      .dd-toolbar button {{
        border: 1px solid #cbd5e1;
        border-radius: 8px;
        background: #ffffff;
        padding: 5px 10px;
        font-weight: 600;
        cursor: pointer;
      }}
      .dd-tip {{
        font-size: 12px;
        color: #4b5563;
      }}
      .dd-sentences {{
        margin-top: 12px;
        display: grid;
        gap: 8px;
      }}
      .dd-row {{
        border: 1px solid #dbe2ea;
        border-radius: 10px;
        background: #ffffff;
        padding: 8px 10px;
      }}
      .dd-hint-hanzi {{
        font-size: 19px;
        line-height: 1.5;
        color: #111827;
      }}
      .dd-hint-pinyin {{
        font-size: 14px;
        line-height: 1.55;
        color: #1f2937;
      }}
      .dd-hint-english {{
        font-size: 14px;
        line-height: 1.55;
        color: #374151;
        margin-bottom: 4px;
      }}
      .dd-line {{
        line-height: 1.9;
        font-size: 18px;
        color: #111827;
        margin-top: 4px;
      }}
      .dd-line-no {{
        font-weight: 700;
        margin-right: 8px;
      }}
      .dd-slot {{
        display: inline-flex;
        min-width: 84px;
        min-height: 34px;
        padding: 2px 4px;
        margin: 0 2px;
        border: 2px dashed #94a3b8;
        border-radius: 8px;
        vertical-align: middle;
        align-items: center;
        justify-content: center;
        background: #ffffff;
      }}
      .dd-slot.over {{
        border-color: #0f766e;
        background: #f0fdfa;
      }}
      .dd-slot.correct {{
        border-color: #16a34a;
        background: #f0fdf4;
      }}
      .dd-slot.wrong {{
        border-color: #ef4444;
        background: #fff1f2;
      }}
      .dd-bank-label {{
        margin-top: 12px;
        margin-bottom: 4px;
        font-size: 14px;
        font-weight: 700;
      }}
      .dd-bank {{
        display: flex;
        gap: 8px;
        flex-wrap: wrap;
        min-height: 44px;
        padding: 8px;
        border: 1px solid #dbe2ea;
        border-radius: 10px;
        background: #ffffff;
      }}
      .dd-token {{
        display: inline-flex;
        align-items: center;
        justify-content: center;
        padding: 6px 10px;
        border: 1px solid #94a3b8;
        border-radius: 8px;
        background: #eff6ff;
        font-weight: 700;
        color: #0f172a;
        cursor: grab;
        user-select: none;
        white-space: nowrap;
      }}
      .dd-token:active {{
        cursor: grabbing;
      }}
      .dd-result {{
        margin-top: 10px;
        font-size: 14px;
      }}
      .dd-row-result {{
        margin-top: 5px;
      }}
      .dd-row-result.good {{
        color: #15803d;
      }}
      .dd-row-result.bad {{
        color: #b91c1c;
      }}
      @media (max-width: 680px) {{
        .dd-line {{
          font-size: 16px;
          line-height: 2.0;
        }}
        .dd-slot {{
          min-width: 72px;
          min-height: 32px;
        }}
      }}
    </style>
    <script>
      (() => {{
        const payload = {payload_json};
        const wrap = document.getElementById("dd-wrap-{safe_id}");
        const sentencesEl = document.getElementById("dd-sentences-{safe_id}");
        const bankEl = document.getElementById("dd-bank-{safe_id}");
        const resultEl = document.getElementById("dd-result-{safe_id}");
        const checkBtn = document.getElementById("check-btn-{safe_id}");
        const resetBtn = document.getElementById("reset-btn-{safe_id}");

        const tokenMap = new Map(payload.tokens.map((t) => [t.id, t]));
        const tokenOrder = new Map(payload.tokens.map((t, idx) => [t.id, idx]));
        let bankTokenIds = payload.tokens.map((t) => t.id);
        let slots = Array(payload.expected_slots.length).fill(null);
        let graded = Array(payload.expected_slots.length).fill(null);

        const keepBankOrder = () => {{
          bankTokenIds.sort((a, b) => tokenOrder.get(a) - tokenOrder.get(b));
        }};

        const removeTokenFromCurrentLocation = (tokenId) => {{
          const iBank = bankTokenIds.indexOf(tokenId);
          if (iBank >= 0) {{
            bankTokenIds.splice(iBank, 1);
            return;
          }}
          const iSlot = slots.indexOf(tokenId);
          if (iSlot >= 0) {{
            slots[iSlot] = null;
          }}
        }};

        const sendToBank = (tokenId) => {{
          if (!tokenId) return;
          removeTokenFromCurrentLocation(tokenId);
          bankTokenIds.push(tokenId);
          keepBankOrder();
        }};

        const placeInSlot = (slotIndex, tokenId) => {{
          if (!tokenMap.has(tokenId)) return;
          removeTokenFromCurrentLocation(tokenId);
          const replaced = slots[slotIndex];
          if (replaced) {{
            bankTokenIds.push(replaced);
          }}
          slots[slotIndex] = tokenId;
          graded = Array(payload.expected_slots.length).fill(null);
          keepBankOrder();
          render();
        }};

        const clearSlot = (slotIndex) => {{
          const existing = slots[slotIndex];
          if (!existing) return;
          slots[slotIndex] = null;
          bankTokenIds.push(existing);
          graded = Array(payload.expected_slots.length).fill(null);
          keepBankOrder();
          render();
        }};

        const makeTokenEl = (token, slotIndex = null) => {{
          const tokenEl = document.createElement("div");
          tokenEl.className = "dd-token";
          tokenEl.textContent = token.text;
          tokenEl.draggable = true;
          tokenEl.dataset.tokenId = token.id;
          tokenEl.addEventListener("dragstart", (evt) => {{
            evt.dataTransfer.setData("text/plain", token.id);
          }});
          if (slotIndex !== null) {{
            tokenEl.title = "Click to return to word bank";
            tokenEl.addEventListener("click", () => clearSlot(slotIndex));
          }}
          return tokenEl;
        }};

        const render = () => {{
          sentencesEl.innerHTML = "";
          bankEl.innerHTML = "";

          payload.sentences.forEach((sentence, rowIdx) => {{
            const rowWrap = document.createElement("div");
            rowWrap.className = "dd-row";

            if (sentence.hanzi_hint) {{
              const hanziHint = document.createElement("div");
              hanziHint.className = "dd-hint-hanzi";
              hanziHint.textContent = sentence.hanzi_hint;
              rowWrap.appendChild(hanziHint);
            }}

            const line = document.createElement("div");
            line.className = "dd-line";

            const no = document.createElement("span");
            no.className = "dd-line-no";
            no.textContent = String(rowIdx + 1) + ".";
            line.appendChild(no);

            for (let i = 0; i < sentence.segments.length; i++) {{
              const textSpan = document.createElement("span");
              textSpan.textContent = sentence.segments[i];
              line.appendChild(textSpan);

              if (i < sentence.slot_indexes.length) {{
                const slotIndex = sentence.slot_indexes[i];
                const slotEl = document.createElement("span");
                slotEl.className = "dd-slot";
                slotEl.dataset.slotIndex = String(slotIndex);
                if (graded[slotIndex] === true) {{
                  slotEl.classList.add("correct");
                }} else if (graded[slotIndex] === false) {{
                  slotEl.classList.add("wrong");
                }}

                slotEl.addEventListener("dragover", (evt) => {{
                  evt.preventDefault();
                  slotEl.classList.add("over");
                }});
                slotEl.addEventListener("dragleave", () => {{
                  slotEl.classList.remove("over");
                }});
                slotEl.addEventListener("drop", (evt) => {{
                  evt.preventDefault();
                  slotEl.classList.remove("over");
                  const tokenId = evt.dataTransfer.getData("text/plain");
                  placeInSlot(slotIndex, tokenId);
                }});

                const tokenId = slots[slotIndex];
                if (tokenId && tokenMap.has(tokenId)) {{
                  slotEl.appendChild(makeTokenEl(tokenMap.get(tokenId), slotIndex));
                }}
                line.appendChild(slotEl);
              }}
            }}

            rowWrap.appendChild(line);
            if (sentence.pinyin_hint) {{
              const pinyinHint = document.createElement("div");
              pinyinHint.className = "dd-hint-pinyin";
              pinyinHint.textContent = sentence.pinyin_hint;
              rowWrap.appendChild(pinyinHint);
            }}
            if (sentence.english_hint) {{
              const englishHint = document.createElement("div");
              englishHint.className = "dd-hint-english";
              englishHint.textContent = sentence.english_hint;
              rowWrap.appendChild(englishHint);
            }}
            sentencesEl.appendChild(rowWrap);
          }});

          keepBankOrder();
          bankTokenIds.forEach((tokenId) => {{
            const token = tokenMap.get(tokenId);
            if (token) {{
              bankEl.appendChild(makeTokenEl(token, null));
            }}
          }});
        }};

        const checkAnswers = () => {{
          let score = 0;
          const details = [];

          for (let i = 0; i < slots.length; i++) {{
            const tokenId = slots[i];
            const given = tokenId ? String(tokenMap.get(tokenId).text).trim() : "";
            const expected = payload.expected_slots[i] || [];
            const ok = expected.includes(given);
            graded[i] = ok;
            if (ok) {{
              score += 1;
            }}
            details.push({{
              slot: i + 1,
              ok,
              given: given || "(empty)",
              expected: expected.length ? expected.join(" / ") : "(none)"
            }});
          }}

          render();

          const total = slots.length;
          const pct = total > 0 ? Math.round((score / total) * 100) : 0;
          const summary = document.createElement("div");
          summary.innerHTML = "<strong>Score: " + score + "/" + total + " (" + pct + "%)</strong>";

          const rows = document.createElement("div");
          details.forEach((d) => {{
            const row = document.createElement("div");
            row.className = "dd-row-result " + (d.ok ? "good" : "bad");
            row.textContent = "Blank " + d.slot + ": " + (d.ok ? "Correct" : ("Try again | Your answer: " + d.given + " | Expected: " + d.expected));
            rows.appendChild(row);
          }});

          resultEl.innerHTML = "";
          resultEl.appendChild(summary);
          resultEl.appendChild(rows);
        }};

        checkBtn.addEventListener("click", checkAnswers);
        resetBtn.addEventListener("click", () => {{
          bankTokenIds = payload.tokens.map((t) => t.id);
          slots = Array(payload.expected_slots.length).fill(null);
          graded = Array(payload.expected_slots.length).fill(null);
          resultEl.innerHTML = "";
          render();
        }});

        render();
      }})();
    </script>
    """

    if height is None:
        estimated = 300 + int(len(sentences) * 78)
        height = max(560, min(estimated, 1700))
    components.html(html, height=int(height), scrolling=True)


def render_stroke_checker(char: str):
    safe_char = json.dumps(char, ensure_ascii=False)
    html = f"""
    <div class="stroke-wrap">
      <div class="stroke-head">Stroke Check Practice</div>
      <div class="stroke-tip">Write the character in the correct stroke order and direction. The checker marks each stroke when it is correct.</div>
      <div id="stroke-board"></div>
      <div class="stroke-controls">
        <button id="start-quiz" type="button">Start Checking</button>
        <button id="restart-quiz" type="button">Restart</button>
        <button id="show-outline" type="button">Show Outline</button>
        <button id="hide-outline" type="button">Hide Outline</button>
      </div>
      <div id="stroke-status" class="stroke-status"></div>
    </div>
    <style>
      .stroke-wrap {{
        border: 1px solid #dbe2ea;
        border-radius: 12px;
        background: #f9fbfd;
        padding: 12px;
      }}
      .stroke-head {{
        font-size: 16px;
        font-weight: 700;
      }}
      .stroke-tip {{
        margin-top: 4px;
        margin-bottom: 10px;
        font-size: 13px;
        color: #4b5563;
      }}
      #stroke-board {{
        width: 100%;
        min-height: 360px;
        border: 2px solid #cbd5e1;
        border-radius: 10px;
        background: #ffffff;
      }}
      .stroke-controls {{
        margin-top: 10px;
        display: flex;
        gap: 8px;
        flex-wrap: wrap;
      }}
      .stroke-controls button {{
        border: 1px solid #cbd5e1;
        border-radius: 8px;
        background: #ffffff;
        padding: 6px 10px;
        font-weight: 600;
        cursor: pointer;
      }}
      .stroke-status {{
        margin-top: 10px;
        font-size: 14px;
        color: #0f172a;
      }}
    </style>
    <script src="https://cdn.jsdelivr.net/npm/hanzi-writer@3.7/dist/hanzi-writer.min.js"></script>
    <script>
      (() => {{
        const character = {safe_char};
        const boardId = "stroke-board";
        const statusEl = document.getElementById("stroke-status");
        const startBtn = document.getElementById("start-quiz");
        const restartBtn = document.getElementById("restart-quiz");
        const showOutlineBtn = document.getElementById("show-outline");
        const hideOutlineBtn = document.getElementById("hide-outline");
        const boardSize = Math.max(260, Math.min(420, Math.floor(window.innerWidth - 36)));

        let totalStrokes = null;
        let doneStrokes = 0;
        let mistakes = 0;
        let writer = null;

        const renderStatus = (extra = "") => {{
          const total = totalStrokes ?? "?";
          const base = "Progress: " + doneStrokes + "/" + total + " strokes | Mistakes: " + mistakes;
          statusEl.textContent = extra ? (base + " | " + extra) : base;
        }};

        const startQuiz = () => {{
          doneStrokes = 0;
          mistakes = 0;
          renderStatus("Checking started");
          writer.quiz({{
            showHintAfterMisses: 2,
            highlightOnComplete: true,
            onCorrectStroke: (data) => {{
              doneStrokes = data.strokeNum + 1;
              mistakes = data.totalMistakes;
              renderStatus("Stroke " + (data.strokeNum + 1) + " correct");
            }},
            onMistake: (data) => {{
              doneStrokes = data.strokeNum;
              mistakes = data.totalMistakes;
              renderStatus("Mistake on stroke " + (data.strokeNum + 1));
            }},
            onComplete: (summary) => {{
              doneStrokes = totalStrokes ?? doneStrokes;
              mistakes = summary.totalMistakes;
              renderStatus("Completed");
            }}
          }});
        }};

        writer = HanziWriter.create(boardId, character, {{
          width: boardSize,
          height: boardSize,
          padding: 16,
          showCharacter: false,
          showOutline: true,
          strokeColor: "#111827",
          outlineColor: "#cbd5e1",
          drawingColor: "#0f766e",
          radicalColor: "#2563eb",
          onLoadCharDataSuccess: (charData) => {{
            totalStrokes = charData.strokes.length;
            doneStrokes = 0;
            mistakes = 0;
            renderStatus("Ready");
            startQuiz();
          }},
          onLoadCharDataError: () => {{
            statusEl.textContent = "Unable to load stroke checking data for this character.";
          }}
        }});

        startBtn.addEventListener("click", startQuiz);
        restartBtn.addEventListener("click", startQuiz);
        showOutlineBtn.addEventListener("click", () => writer.showOutline({{ duration: 120 }}));
        hideOutlineBtn.addEventListener("click", () => writer.hideOutline({{ duration: 120 }}));
      }})();
    </script>
    """
    components.html(html, height=620, scrolling=False)


def parse_official_hsk_level(levels_text: str):
    if not isinstance(levels_text, str) or not levels_text.strip():
        return None
    matches = re.findall(r"(?:newest|new|old)-(\d+)", levels_text)
    if not matches:
        return None
    nums = sorted({int(m) for m in matches if m.isdigit()})
    if not nums:
        return None
    # Use the easiest observed level as entry point for learners.
    return nums[0]


@st.cache_data(show_spinner=False)
def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing data file: {path}")

    df = pd.read_csv(path, encoding="utf-8-sig")

    df["frequency_rank"] = pd.to_numeric(df["frequency_rank"], errors="coerce")
    max_freq = df["frequency_rank"].dropna().max()
    if pd.isna(max_freq):
        max_freq = 999999
    df["frequency_rank"] = df["frequency_rank"].fillna(max_freq + 100000)

    df["official_hsk"] = df["hsk_levels"].apply(parse_official_hsk_level)

    # Custom proficiency levels 1-10 based on usage frequency.
    rank_order = df["frequency_rank"].rank(method="first", ascending=True)
    df["proficiency_10"] = pd.qcut(rank_order, 10, labels=list(range(1, 11))).astype(int)
    df["pinyin_key"] = df["pinyin"].fillna("").astype(str).apply(normalize_pinyin)

    def length_label(n: int) -> str:
        if n == 1:
            return "1 char"
        if n == 2:
            return "2 chars"
        if n == 3:
            return "3 chars"
        return "4+ chars"

    df["length_label"] = df["character_count"].fillna(0).astype(int).apply(length_label)
    return df


try:
    vocab = load_data(DATA_PATH)
except Exception as exc:
    st.error(f"Could not load vocabulary data: {exc}")
    st.stop()

st.title("Mandarin Proficiency Trainer")
st.caption(
    "English-support learning for intermediate learners: single Hanzi, 2-character words, 3+ chunks, daily dialogue, and quiz practice."
)

with st.sidebar:
    st.header("Study Filters")

    level_10 = st.multiselect(
        "Proficiency Level (1-10)",
        options=list(range(1, 11)),
        default=list(range(1, 11)),
        help="Custom levels 1-10. Level 1 is most frequent/easy-use words; level 10 is least frequent/most advanced.",
    )

    official_levels = sorted([int(x) for x in vocab["official_hsk"].dropna().unique()])
    selected_official = st.multiselect(
        "Official HSK Levels Found",
        options=official_levels,
        default=official_levels,
        help="From the source tags (old/new/newest standards).",
    )

    length_options = ["1 char", "2 chars", "3 chars", "4+ chars"]
    selected_lengths = st.multiselect(
        "Character Length",
        options=length_options,
        default=length_options,
    )

    keyword = st.text_input("Search Hanzi / Pinyin / English", value="").strip()
    family_query = st.text_input(
        "Character Family Search (e.g. 新 or xin)",
        value="",
        help="Shows same-character/same-syllable families, ordered by 1 char -> 2 chars -> 3+ chars.",
    ).strip()

    sort_mode = st.selectbox(
        "Sort By",
        options=["Alphabetical (Pinyin A-Z)", "Alphabetical (Hanzi)", "Frequency (common first)", "Proficiency Level"],
        index=0,
    )

    show_all_rows = st.checkbox("Show all filtered rows", value=True)
    if show_all_rows:
        max_rows = None
    else:
        max_rows = st.number_input(
            "Rows to show",
            min_value=20,
            max_value=int(len(vocab)),
            value=min(200, int(len(vocab))),
            step=20,
        )

filtered = vocab.copy()

if level_10:
    filtered = filtered[filtered["proficiency_10"].isin(level_10)]
if selected_official:
    filtered = filtered[filtered["official_hsk"].isin(selected_official)]
if selected_lengths:
    filtered = filtered[filtered["length_label"].isin(selected_lengths)]

if keyword:
    mask = (
        filtered["hanzi_simplified"].astype(str).str.contains(keyword, case=False, na=False)
        | filtered["pinyin"].astype(str).str.contains(keyword, case=False, na=False)
        | filtered["english_meanings"].astype(str).str.contains(keyword, case=False, na=False)
    )
    filtered = filtered[mask]

if family_query:
    has_hanzi = bool(re.search(r"[\u4e00-\u9fff]", family_query))

    if has_hanzi:
        family_char = next((ch for ch in family_query if re.match(r"[\u4e00-\u9fff]", ch)), "")
        if family_char:
            mask = filtered["hanzi_simplified"].astype(str).str.contains(re.escape(family_char), regex=True, na=False)
            filtered = filtered[mask].copy()
            filtered["family_priority"] = 3
            filtered.loc[filtered["hanzi_simplified"] == family_char, "family_priority"] = 0
            filtered.loc[
                (filtered["hanzi_simplified"] != family_char)
                & filtered["hanzi_simplified"].astype(str).str.startswith(family_char, na=False),
                "family_priority",
            ] = 1
            filtered.loc[
                (filtered["family_priority"] == 3)
                & filtered["hanzi_simplified"].astype(str).str.contains(re.escape(family_char), regex=True, na=False),
                "family_priority",
            ] = 2
    else:
        query_key = normalize_pinyin(family_query)
        if query_key:
            pinyin_tokens = filtered["pinyin_key"].fillna("")

            exact_single = (filtered["character_count"] == 1) & (pinyin_tokens == query_key)
            starts_with_token = pinyin_tokens.str.startswith(query_key + " ", na=False) | (pinyin_tokens == query_key)
            token_contains = pinyin_tokens.str.contains(rf"(^|\s){re.escape(query_key)}($|\s)", regex=True, na=False)
            partial_contains = pinyin_tokens.str.contains(re.escape(query_key), regex=True, na=False)

            mask = exact_single | starts_with_token | token_contains | partial_contains
            filtered = filtered[mask].copy()
            filtered["family_priority"] = 4
            filtered.loc[partial_contains, "family_priority"] = 3
            filtered.loc[token_contains, "family_priority"] = 2
            filtered.loc[starts_with_token, "family_priority"] = 1
            filtered.loc[exact_single, "family_priority"] = 0

if family_query and "family_priority" in filtered.columns:
    filtered = filtered.sort_values(
        by=["family_priority", "character_count", "pinyin_key", "hanzi_simplified"],
        ascending=[True, True, True, True],
    )
elif sort_mode == "Alphabetical (Pinyin A-Z)":
    filtered = filtered.sort_values(by=["pinyin_key", "character_count", "hanzi_simplified"], ascending=[True, True, True])
elif sort_mode == "Alphabetical (Hanzi)":
    filtered = filtered.sort_values(by=["hanzi_simplified", "pinyin_key"], ascending=[True, True])
elif sort_mode == "Frequency (common first)":
    filtered = filtered.sort_values(by=["frequency_rank", "character_count", "hanzi_simplified"], ascending=[True, True, True])
else:
    filtered = filtered.sort_values(by=["proficiency_10", "frequency_rank"], ascending=[True, True])

tab_vocab, tab_stroke, tab_convo, tab_quiz, tab_cue = st.tabs(
    ["Vocabulary Explorer", "Stroke Order Practice", "Daily Conversation", "Quick Quiz", "Cue Cards"]
)

with tab_vocab:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Filtered Words", f"{len(filtered):,}")
    col2.metric("1-char", f"{(filtered['character_count'] == 1).sum():,}")
    col3.metric("2-char", f"{(filtered['character_count'] == 2).sum():,}")
    col4.metric("3+ char", f"{(filtered['character_count'] >= 3).sum():,}")
    st.caption(f"Vocabulary dataset loaded: {len(vocab):,} total words (no 500-word cap).")

    st.markdown("### Single -> Two -> Three+ Flow")
    st.write(
        "Start with single Hanzi meaning, then jump to 2-character combinations, then 3+ chunks for natural daily fluency."
    )

    display_cols = [
        "hanzi_simplified",
        "pinyin",
        "english_meanings",
        "character_count",
        "proficiency_10",
        "official_hsk",
        "example_zh",
        "example_en",
    ]

    if max_rows is None:
        shown_base = filtered[display_cols]
    else:
        shown_base = filtered[display_cols].head(int(max_rows))

    shown = shown_base.rename(
        columns={
            "hanzi_simplified": "Hanzi (Simplified)",
            "pinyin": "Pinyin",
            "english_meanings": "English",
            "character_count": "Chars",
            "proficiency_10": "Level 1-10",
            "official_hsk": "HSK",
            "example_zh": "Example (中文)",
            "example_en": "Example (EN)",
        }
    )
    st.dataframe(shown, width="stretch", hide_index=True)

    csv_bytes = filtered.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "Download Filtered Vocabulary CSV",
        data=csv_bytes,
        file_name="mandarin_filtered_vocab.csv",
        mime="text/csv",
    )

with tab_stroke:
    st.markdown("### Stroke Order Writing Exercise (with Live Checking)")
    st.caption("Pick one Hanzi, then write it directly. The checker validates each stroke as you draw.")

    single_hanzi_df = (
        filtered[filtered["character_count"] == 1][["hanzi_simplified", "pinyin", "english_meanings", "pinyin_key"]]
        .drop_duplicates(subset=["hanzi_simplified"])
        .sort_values(by=["pinyin_key", "hanzi_simplified"])
    )
    single_hanzi_list = single_hanzi_df["hanzi_simplified"].tolist()

    if "stroke_char" not in st.session_state:
        st.session_state["stroke_char"] = single_hanzi_list[0] if single_hanzi_list else "你"

    c1, c2 = st.columns([2, 1])
    with c1:
        if single_hanzi_list:
            picked = st.selectbox("Choose a Hanzi from current filter", options=single_hanzi_list, key="stroke_pick")
        else:
            picked = "你"
    with c2:
        custom_input = st.text_input("Or type a Hanzi", value=st.session_state["stroke_char"], key="stroke_custom")

    target_char = extract_first_hanzi(custom_input) or extract_first_hanzi(picked) or "你"
    st.session_state["stroke_char"] = target_char

    stroke_data = fetch_stroke_svg_data(target_char)
    if not stroke_data.get("ok"):
        st.warning(stroke_data.get("error", "No stroke data available."))
    else:
        st.markdown(f"**Character:** {target_char}  |  **Unicode:** U+{stroke_data['codepoint']:04X}")
        st.markdown(f"**Stroke count:** {stroke_data['stroke_count']}")
        if stroke_data["sequence"]:
            seq_text = " -> ".join(str(n) for n in stroke_data["sequence"])
            st.markdown(f"**Stroke sequence:** {seq_text}")

        render_stroke_checker(target_char)
        st.link_button("Open stroke source (SVG)", stroke_data["source_url"])

        related = filtered[filtered["hanzi_simplified"].astype(str).str.contains(re.escape(target_char), regex=True, na=False)].copy()
        related = related.sort_values(by=["character_count", "pinyin_key", "hanzi_simplified"])
        st.markdown("#### Related Words by Character Length")
        for n in [1, 2, 3]:
            sub = related[related["character_count"] == n][["hanzi_simplified", "pinyin", "english_meanings"]].head(12)
            label = "1 char" if n == 1 else f"{n} chars"
            if not sub.empty:
                st.markdown(f"**{label}**")
                st.dataframe(
                    sub.rename(
                        columns={
                            "hanzi_simplified": "Hanzi",
                            "pinyin": "Pinyin",
                            "english_meanings": "English",
                        }
                    ),
                    width="stretch",
                    hide_index=True,
                )
        sub4 = related[related["character_count"] >= 4][["hanzi_simplified", "pinyin", "english_meanings"]].head(12)
        if not sub4.empty:
            st.markdown("**4+ chars**")
            st.dataframe(
                sub4.rename(
                    columns={
                        "hanzi_simplified": "Hanzi",
                        "pinyin": "Pinyin",
                        "english_meanings": "English",
                    }
                ),
                width="stretch",
                hide_index=True,
            )

with tab_convo:
    st.markdown("### Daily Conversation")
    convo_ref_tab, convo_fill_tab = st.tabs(["Reference (Hanzi + Pinyin + English)", "Drag & Drop Practice"])

    with convo_ref_tab:
        expression_lookup = _build_expression_lookup(vocab)

        conversations = [
            {
                "title": "1) Work Conversation",
                "lines": [
                    ("A", "你现在方便聊两分钟吗？", "Nǐ xiànzài fāngbiàn liáo liǎng fēnzhōng ma?", "Are you free to chat for two minutes right now?"),
                    ("B", "可以，我刚开完会。", "Kěyǐ, wǒ gāng kāiwán huì.", "Sure, I just finished a meeting."),
                    ("A", "客户希望我们把截止日期提前到周五。", "Kèhù xīwàng wǒmen bǎ jiézhǐ rìqī tíqián dào zhōuwǔ.", "The client wants us to move the deadline up to Friday."),
                    ("B", "那我先改时间表，晚点发你新版本。", "Nà wǒ xiān gǎi shíjiānbiǎo, wǎndiǎn fā nǐ xīn bǎnběn.", "Then I'll update the timeline first and send you a new version later."),
                ],
            },
            {
                "title": "2) Commute Conversation",
                "lines": [
                    ("A", "今天路上堵不堵？", "Jīntiān lùshang dǔ bu dǔ?", "Is traffic bad today?"),
                    ("B", "高架有点慢，我改坐地铁了。", "Gāojià yǒudiǎn màn, wǒ gǎi zuò dìtiě le.", "The elevated road is slow, so I switched to the subway."),
                    ("A", "那我们在公司楼下咖啡店见吧。", "Nà wǒmen zài gōngsī lóuxià kāfēidiàn jiàn ba.", "Then let's meet at the coffee shop under the office building."),
                    ("B", "好，我大概八点二十到。", "Hǎo, wǒ dàgài bā diǎn èrshí dào.", "Great, I'll arrive around 8:20."),
                ],
            },
            {
                "title": "3) Weekend Plan Conversation",
                "lines": [
                    ("A", "周末你想在家休息还是出去走走？", "Zhōumò nǐ xiǎng zài jiā xiūxi háishi chūqù zǒuzou?", "Do you want to rest at home this weekend or go out?"),
                    ("B", "我想先去菜市场，然后下午去看展览。", "Wǒ xiǎng xiān qù càishìchǎng, ránhòu xiàwǔ qù kàn zhǎnlǎn.", "I want to go to the market first, then see an exhibition in the afternoon."),
                    ("A", "听起来不错，我可以一起吗？", "Tīng qǐlái búcuò, wǒ kěyǐ yìqǐ ma?", "Sounds great. Can I join?"),
                    ("B", "当然可以，下午两点地铁站见。", "Dāngrán kěyǐ, xiàwǔ liǎng diǎn dìtiězhàn jiàn.", "Of course. Let's meet at the subway station at 2 p.m."),
                ],
            },
        ]

        for convo_idx, convo in enumerate(conversations):
            st.markdown(f"#### {convo['title']}")
            for line_idx, (role, hanzi, pinyin, english) in enumerate(convo["lines"]):
                render_conversation_line_selector(
                    activity_id=f"convo_inline_{convo_idx}_{line_idx}",
                    role=role,
                    hanzi=hanzi,
                    pinyin=pinyin,
                    english=english,
                    expression_lookup=expression_lookup,
                )
            st.divider()

    with convo_fill_tab:
        daily_mode = st.radio(
            "Practice display mode",
            options=["With Hanzi", "No Hanzi (Pinyin only)"],
            horizontal=True,
            key="daily_drag_mode",
        )
        use_hanzi_mode = daily_mode == "With Hanzi"

        work_word_bank = (
            ["有空", "怎么了", "问", "怎么样", "可能", "差不多了", "一起", "当然"]
            if use_hanzi_mode
            else ["yǒu kòng", "zěnme le", "wèn", "zěnmeyàng", "kěnéng", "chàbuduō le", "yìqǐ", "dāngrán"]
        )
        work_sentences = [
            {
                "text": "你现在 ___ 吗？" if use_hanzi_mode else "Nǐ xiànzài ___ ma?",
                "answers": ["有空"] if use_hanzi_mode else ["yǒu kòng"],
                "pinyin_hint": "Nǐ xiànzài ___ ma?" if use_hanzi_mode else "",
                "english_hint": "Are you free now?",
            },
            {
                "text": "有，___？" if use_hanzi_mode else "Yǒu, ___?",
                "answers": ["怎么了"] if use_hanzi_mode else ["zěnme le"],
                "pinyin_hint": "Yǒu, ___?" if use_hanzi_mode else "",
                "english_hint": "Yes, what's up?",
            },
            {
                "text": "我想 ___ 一下这个计划 ___。" if use_hanzi_mode else "Wǒ xiǎng ___ yíxià zhège jìhuà ___.",
                "answers": ["问", "怎么样"] if use_hanzi_mode else ["wèn", "zěnmeyàng"],
                "pinyin_hint": "Wǒ xiǎng ___ yíxià zhège jìhuà ___." if use_hanzi_mode else "",
                "english_hint": "I want to ask how this plan looks.",
            },
            {
                "text": "___ 要改一点，不过 ___。" if use_hanzi_mode else "___ yào gǎi yìdiǎn, búguò ___.",
                "answers": ["可能", "差不多了"] if use_hanzi_mode else ["kěnéng", "chàbuduō le"],
                "pinyin_hint": "___ yào gǎi yìdiǎn, búguò ___." if use_hanzi_mode else "",
                "english_hint": "It may need some changes, but it is almost done.",
            },
        ]
        render_drag_drop_activity(
            activity_id="daily_work_fill",
            title="Work Dialogue Practice",
            prompt="Drag each word into the correct blank to rebuild the conversation.",
            word_bank=work_word_bank,
            sentences=work_sentences,
        )
        st.divider()
        commute_word_bank = (
            ["怎么", "地铁", "有时候", "公共汽车", "一起", "可以", "地铁站", "八点", "公司"]
            if use_hanzi_mode
            else ["zěnme", "dìtiě", "yǒu shíhou", "gōnggòng qìchē", "yìqǐ", "kěyǐ", "dìtiězhàn", "bā diǎn", "gōngsī"]
        )
        commute_sentences = [
            {
                "text": "你 ___ 去公司？" if use_hanzi_mode else "Nǐ ___ qù gōngsī?",
                "answers": ["怎么"] if use_hanzi_mode else ["zěnme"],
                "pinyin_hint": "Nǐ ___ qù gōngsī?" if use_hanzi_mode else "",
                "english_hint": "How do you get to the office?",
            },
            {
                "text": "我坐 ___，___ 坐 ___。" if use_hanzi_mode else "Wǒ zuò ___, ___ zuò ___.",
                "answers": ["地铁", "有时候", "公共汽车"] if use_hanzi_mode else ["dìtiě", "yǒu shíhou", "gōnggòng qìchē"],
                "pinyin_hint": "Wǒ zuò ___, ___ zuò ___." if use_hanzi_mode else "",
                "english_hint": "I take the subway, and sometimes the bus.",
            },
            {
                "text": "明天 ___ 去吗？" if use_hanzi_mode else "Míngtiān ___ qù ma?",
                "answers": ["一起"] if use_hanzi_mode else ["yìqǐ"],
                "pinyin_hint": "Míngtiān ___ qù ma?" if use_hanzi_mode else "",
                "english_hint": "Do you want to go together tomorrow?",
            },
            {
                "text": "___，___ ___ 见。" if use_hanzi_mode else "___, ___ ___ jiàn.",
                "answers": ["可以", "地铁站", "八点"] if use_hanzi_mode else ["kěyǐ", "dìtiězhàn", "bā diǎn"],
                "pinyin_hint": "___, ___ ___ jiàn." if use_hanzi_mode else "",
                "english_hint": "Sure, see you at the station at 8.",
            },
        ]
        render_drag_drop_activity(
            activity_id="daily_commute_fill",
            title="Commute Dialogue Practice",
            prompt="Drag and drop each phrase into the best spot.",
            word_bank=commute_word_bank,
            sentences=commute_sentences,
        )
        st.divider()
        food_word_bank = (
            ["吃饭", "还没有", "来不及", "早饭", "先", "餐厅", "不客气", "请"]
            if use_hanzi_mode
            else ["chīfàn", "hái méiyǒu", "lái bu jí", "zǎofàn", "xiān", "cāntīng", "bú kèqi", "qǐng"]
        )
        food_sentences = [
            {
                "text": "你 ___ 了吗？" if use_hanzi_mode else "Nǐ ___ le ma?",
                "answers": ["吃饭"] if use_hanzi_mode else ["chīfàn"],
                "pinyin_hint": "Nǐ ___ le ma?" if use_hanzi_mode else "",
                "english_hint": "Have you eaten?",
            },
            {
                "text": "___，___ 吃 ___。" if use_hanzi_mode else "___, ___ chī ___.",
                "answers": ["还没有", "来不及", "早饭"] if use_hanzi_mode else ["hái méiyǒu", "lái bu jí", "zǎofàn"],
                "pinyin_hint": "___, ___ chī ___." if use_hanzi_mode else "",
                "english_hint": "Not yet, I didn't have time to eat breakfast.",
            },
            {
                "text": "那我们 ___ 去 ___ 吧。" if use_hanzi_mode else "Nà wǒmen ___ qù ___ ba.",
                "answers": ["先", "餐厅"] if use_hanzi_mode else ["xiān", "cāntīng"],
                "pinyin_hint": "Nà wǒmen ___ qù ___ ba." if use_hanzi_mode else "",
                "english_hint": "Then let's go to a restaurant first.",
            },
            {
                "text": "好啊，___，我 ___ 你。" if use_hanzi_mode else "Hǎo a, ___, wǒ ___ nǐ.",
                "answers": ["不客气", "请"] if use_hanzi_mode else ["bú kèqi", "qǐng"],
                "pinyin_hint": "Hǎo a, ___, wǒ ___ nǐ." if use_hanzi_mode else "",
                "english_hint": "Sounds good, no worries, I'll treat you.",
            },
        ]
        render_drag_drop_activity(
            activity_id="daily_food_fill",
            title="Food + Social Dialogue Practice",
            prompt="Drag the word bank items into each blank naturally.",
            word_bank=food_word_bank,
            sentences=food_sentences,
        )

with tab_quiz:
    st.markdown("### Quick Quiz")
    st.caption("Choose quiz mode, prompt style, and question count.")

    quiz_mode = st.radio(
        "Quiz mode",
        options=["Drag & Drop Cloze", "English Translation"],
        horizontal=True,
    )
    prompt_mode = st.radio(
        "Prompt display",
        options=["Hanzi + Pinyin", "No Pinyin (English + Hanzi hint)"],
        horizontal=True,
        key="quiz_prompt_mode",
    )
    prompt_style = "hanzi_pinyin" if prompt_mode == "Hanzi + Pinyin" else "no_pinyin_en_hanzi_hint"
    question_count = st.slider("Number of questions", min_value=8, max_value=60, value=24, step=4)

    if quiz_mode == "Drag & Drop Cloze":
        sentences, word_bank = _prepare_cloze_quiz_items(
            df=filtered,
            question_count=question_count,
            prompt_style=prompt_style,
        )
        if not sentences or not word_bank:
            st.warning("Not enough rows for dynamic cloze quiz with current filters.")
        else:
            cloze_prompt = (
                "Drag each item to the blank. Hanzi sentence is shown with pinyin support below each line. Word bank is split per page."
                if prompt_style == "hanzi_pinyin"
                else "Drag each item to the blank. This no-pinyin mode uses the exact same Hanzi questions, just without the pinyin line. Word bank is split per page."
            )
            render_paginated_drag_drop_activity(
                activity_id="quiz_dynamic_cloze",
                title="Dynamic Cloze Quiz",
                prompt=cloze_prompt,
                sentences=sentences,
                word_bank=word_bank,
                page_size=8,
            )
    else:
        render_translation_quiz(
            activity_id="quiz_translation",
            df=filtered,
            question_count=question_count,
            prompt_style=prompt_style,
        )

with tab_cue:
    st.markdown("### Cue Cards")
    st.caption("Front: Hanzi + pinyin + meaning. Back: usage sentence + sentence pinyin + English translation.")

    cue_cols = ["hanzi_simplified", "pinyin", "english_meanings", "example_zh", "example_en"]
    cue_source = _prioritize_varied_examples(filtered)
    cue_df = cue_source[cue_cols].copy()
    cue_df = cue_df.dropna(subset=["hanzi_simplified"])
    cue_df = cue_df.drop_duplicates(subset=["hanzi_simplified", "pinyin", "english_meanings"])
    cue_df = cue_df.reset_index(drop=True)

    render_cue_card_deck(activity_id="cue_cards_deck", cue_df=cue_df)

st.info(
    "Tip: Keep official HSK filter broad, then use Level 1-10 to progressively increase difficulty from high-frequency daily words to advanced low-frequency words."
)
