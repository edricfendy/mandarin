import re
import unicodedata
import json
import html as html_lib
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

        sentence_blocks.append({"segments": segments, "slot_indexes": slot_indexes})

    tokens = [{"id": f"token_{i}", "text": str(word)} for i, word in enumerate(word_bank)]
    return {"sentences": sentence_blocks, "expected_slots": expected_slots, "tokens": tokens}


def render_drag_drop_activity(activity_id, title, prompt, sentences, word_bank):
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
      .dd-line {{
        line-height: 1.9;
        font-size: 18px;
        color: #111827;
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

            sentencesEl.appendChild(line);
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

    components.html(html, height=720, scrolling=False)


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

        preview_col, draw_col = st.columns([1, 1])
        with preview_col:
            st.image(stroke_data["svg_text"], width=420)
        with draw_col:
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
        conversations = [
            {
                "title": "1) Work Conversation",
                "lines": [
                    ("A", "你现在有空吗？", "Nǐ xiànzài yǒu kòng ma?", "Are you free now?"),
                    ("B", "有，怎么了？", "Yǒu, zěnme le?", "Yes, what's up?"),
                    ("A", "我想问一下这个计划怎么样。", "Wǒ xiǎng wèn yíxià zhège jìhuà zěnmeyàng.", "I want to ask how this plan is."),
                    ("B", "可能要改一点，不过差不多了。", "Kěnéng yào gǎi yìdiǎn, búguò chàbuduō le.", "It may need small changes, but it's almost done."),
                ],
            },
            {
                "title": "2) Commute Conversation",
                "lines": [
                    ("A", "你怎么去公司？", "Nǐ zěnme qù gōngsī?", "How do you go to work?"),
                    ("B", "我坐地铁，有时候坐公共汽车。", "Wǒ zuò dìtiě, yǒu shíhou zuò gōnggòng qìchē.", "I take the subway, sometimes the bus."),
                    ("A", "明天一起去吗？", "Míngtiān yìqǐ qù ma?", "Want to go together tomorrow?"),
                    ("B", "可以，地铁站八点见。", "Kěyǐ, dìtiězhàn bā diǎn jiàn.", "Sure, see you at the subway station at 8."),
                ],
            },
            {
                "title": "3) Social + Food Conversation",
                "lines": [
                    ("A", "你吃饭了吗？", "Nǐ chīfàn le ma?", "Have you eaten?"),
                    ("B", "还没有，来不及吃早饭。", "Hái méiyǒu, lái bu jí chī zǎofàn.", "Not yet, I didn't have time for breakfast."),
                    ("A", "那我们先去餐厅吧。", "Nà wǒmen xiān qù cāntīng ba.", "Then let's go to a restaurant first."),
                    ("B", "好啊，不客气，我请你。", "Hǎo a, bú kèqi, wǒ qǐng nǐ.", "Sounds good. It's my treat."),
                ],
            },
        ]

        for convo in conversations:
            st.markdown(f"#### {convo['title']}")
            st.markdown("| Speaker | Hanzi | Pinyin | English |")
            st.markdown("|---|---|---|---|")
            for role, hanzi, pinyin, english in convo["lines"]:
                st.markdown(f"| {role} | {hanzi} | {pinyin} | {english} |")

    with convo_fill_tab:
        render_drag_drop_activity(
            activity_id="daily_work_fill",
            title="Work Dialogue Practice",
            prompt="Drag each word into the correct blank to rebuild the conversation.",
            word_bank=["有空", "怎么了", "问", "怎么样", "可能", "差不多了", "一起", "当然"],
            sentences=[
                {"text": "你现在 ___ 吗？", "answers": ["有空"]},
                {"text": "有，___？", "answers": ["怎么了"]},
                {"text": "我想 ___ 一下这个计划 ___。", "answers": ["问", "怎么样"]},
                {"text": "___ 要改一点，不过 ___。", "answers": ["可能", "差不多了"]},
            ],
        )
        st.divider()
        render_drag_drop_activity(
            activity_id="daily_commute_fill",
            title="Commute Dialogue Practice",
            prompt="Drag and drop each phrase into the best spot.",
            word_bank=["怎么", "地铁", "有时候", "公共汽车", "一起", "可以", "地铁站", "八点", "公司"],
            sentences=[
                {"text": "你 ___ 去公司？", "answers": ["怎么"]},
                {"text": "我坐 ___，___ 坐 ___。", "answers": ["地铁", "有时候", "公共汽车"]},
                {"text": "明天 ___ 去吗？", "answers": ["一起"]},
                {"text": "___，___ ___ 见。", "answers": ["可以", "地铁站", "八点"]},
            ],
        )
        st.divider()
        render_drag_drop_activity(
            activity_id="daily_food_fill",
            title="Food + Social Dialogue Practice",
            prompt="Drag the word bank items into each blank naturally.",
            word_bank=["吃饭", "还没有", "来不及", "早饭", "先", "餐厅", "不客气", "请"],
            sentences=[
                {"text": "你 ___ 了吗？", "answers": ["吃饭"]},
                {"text": "___，___ 吃 ___。", "answers": ["还没有", "来不及", "早饭"]},
                {"text": "那我们 ___ 去 ___ 吧。", "answers": ["先", "餐厅"]},
                {"text": "好啊，___，我 ___ 你。", "answers": ["不客气", "请"]},
            ],
        )

with tab_quiz:
    st.markdown("### Quick Quiz: Drag and Drop Challenge")
    st.caption("Style: drag the word bank into each blank and check your score.")

    render_drag_drop_activity(
        activity_id="quiz_fill_core",
        title="Quiz Set A",
        prompt="Complete all blanks. Focus on high-frequency daily Mandarin patterns.",
        word_bank=["在", "以后", "给", "来得及", "不客气", "一起", "明天", "地铁站", "可能", "晚到"],
        sentences=[
            {"text": "我 ___ 地铁站。", "answers": ["在"]},
            {"text": "我们 ___ 再聊。", "answers": ["以后"]},
            {"text": "我 ___ 你打电话。", "answers": ["给"]},
            {"text": "现在出发还 ___。", "answers": ["来得及"]},
            {"text": "谢谢你。___。", "answers": ["不客气"]},
            {"text": "我们 ___ ___ 一起去吧。", "answers": ["明天", "一起"]},
        ],
    )

    st.divider()

    render_drag_drop_activity(
        activity_id="quiz_fill_plus",
        title="Quiz Set B (Context + Meaning)",
        prompt="Mixed blanks from transport, scheduling, and social phrases.",
        word_bank=["地铁站", "可能", "晚到", "来不及", "电子邮件", "餐厅", "先", "为什么", "怎么样"],
        sentences=[
            {"text": "___ 就在前面。", "answers": ["地铁站"]},
            {"text": "他 ___ 会 ___。", "answers": ["可能", "晚到"]},
            {"text": "我今天 ___ 吃早饭。", "answers": ["来不及"]},
            {"text": "请发 ___ 给我。", "answers": ["电子邮件"]},
            {"text": "那我们 ___ 去 ___ 吧。", "answers": ["先", "餐厅"]},
        ],
    )

with tab_cue:
    st.markdown("### Cue Cards")
    st.caption("Front: Hanzi + pinyin + meaning. Back: usage sentence + English translation.")

    cue_cols = ["hanzi_simplified", "pinyin", "english_meanings", "example_zh", "example_en"]
    cue_df = filtered[cue_cols].copy()
    cue_df = cue_df.dropna(subset=["hanzi_simplified"])
    cue_df = cue_df.drop_duplicates(subset=["hanzi_simplified", "pinyin", "english_meanings"])
    cue_df = cue_df.reset_index(drop=True)

    if cue_df.empty:
        st.warning("No cue cards available for the current filters.")
    else:
        if "cue_card_index" not in st.session_state:
            st.session_state["cue_card_index"] = 0
        if "cue_card_show_back" not in st.session_state:
            st.session_state["cue_card_show_back"] = False

        total_cards = len(cue_df)
        st.session_state["cue_card_index"] = st.session_state["cue_card_index"] % total_cards

        b1, b2, b3, b4 = st.columns([1, 1, 1, 2])
        with b1:
            if st.button("Previous", key="cue_prev"):
                st.session_state["cue_card_index"] = (st.session_state["cue_card_index"] - 1) % total_cards
                st.session_state["cue_card_show_back"] = False
        with b2:
            if st.button("Next", key="cue_next"):
                st.session_state["cue_card_index"] = (st.session_state["cue_card_index"] + 1) % total_cards
                st.session_state["cue_card_show_back"] = False
        with b3:
            if st.button("Random", key="cue_random"):
                st.session_state["cue_card_index"] = int(cue_df.sample(1).index[0])
                st.session_state["cue_card_show_back"] = False
        with b4:
            if st.button("Flip Front/Back", key="cue_flip"):
                st.session_state["cue_card_show_back"] = not st.session_state["cue_card_show_back"]

        idx = st.session_state["cue_card_index"]
        row = cue_df.iloc[idx]

        hanzi = "" if pd.isna(row["hanzi_simplified"]) else str(row["hanzi_simplified"]).strip()
        pinyin = "" if pd.isna(row["pinyin"]) else str(row["pinyin"]).strip()
        meaning = "" if pd.isna(row["english_meanings"]) else str(row["english_meanings"]).strip()
        ex_zh = "" if pd.isna(row["example_zh"]) else str(row["example_zh"]).strip()
        ex_en = "" if pd.isna(row["example_en"]) else str(row["example_en"]).strip()
        hanzi_html = html_lib.escape(hanzi) if hanzi else "—"
        pinyin_html = html_lib.escape(pinyin) if pinyin else "—"
        meaning_html = html_lib.escape(meaning) if meaning else "—"
        ex_zh_html = html_lib.escape(ex_zh) if ex_zh else "No sentence available."
        ex_en_html = html_lib.escape(ex_en) if ex_en else "No English translation available."

        st.caption(f"Card {idx + 1}/{total_cards}")

        if not st.session_state["cue_card_show_back"]:
            st.markdown(
                f"""
                <div style="border:1px solid #dbe2ea;border-radius:14px;padding:22px;background:#f9fbfd;min-height:260px;">
                  <div style="font-size:54px;line-height:1.1;font-weight:700;margin-bottom:10px;">{hanzi_html}</div>
                  <div style="font-size:22px;font-weight:600;color:#1f2937;margin-bottom:10px;">{pinyin_html}</div>
                  <div style="font-size:18px;color:#0f172a;">{meaning_html}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div style="border:1px solid #dbe2ea;border-radius:14px;padding:22px;background:#f8fafc;min-height:260px;">
                  <div style="font-size:15px;font-weight:700;color:#334155;margin-bottom:6px;">How to use it in a sentence</div>
                  <div style="font-size:26px;line-height:1.4;margin-bottom:12px;">{ex_zh_html}</div>
                  <div style="font-size:15px;font-weight:700;color:#334155;margin-bottom:6px;">English translation</div>
                  <div style="font-size:18px;line-height:1.5;">{ex_en_html}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

st.info(
    "Tip: Keep official HSK filter broad, then use Level 1-10 to progressively increase difficulty from high-frequency daily words to advanced low-frequency words."
)
