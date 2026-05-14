import re
import unicodedata
import json
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


def render_word_bank(words):
    chips = " ".join(
        [
            f"<span style='display:inline-block;padding:6px 10px;margin:4px 6px 4px 0;border:1px solid #d1d5db;border-radius:8px;background:#f8fafc;font-weight:600;'>{w}</span>"
            for w in words
        ]
    )
    st.markdown("**Word Bank**")
    st.markdown(chips, unsafe_allow_html=True)


def _is_blank_correct(user_value, expected):
    if isinstance(expected, list):
        return user_value in expected
    return user_value == expected


def _expected_to_text(expected):
    if isinstance(expected, list):
        return " / ".join(expected)
    return expected


def render_fill_blank_activity(activity_id, title, prompt, sentences, word_bank):
    st.markdown(f"#### {title}")
    st.caption(prompt)
    render_word_bank(word_bank)

    with st.form(f"{activity_id}_form"):
        user_picks = []
        expected_vals = []
        blank_no = 1

        for i, sentence in enumerate(sentences, start=1):
            st.markdown(f"**{i}. {sentence['text']}**")
            for expected in sentence["answers"]:
                pick = st.selectbox(
                    f"Blank {blank_no}",
                    options=["(choose)"] + word_bank,
                    key=f"{activity_id}_blank_{blank_no}",
                )
                user_picks.append(pick)
                expected_vals.append(expected)
                blank_no += 1

        submitted = st.form_submit_button("Check Fill-in Answers")

    if submitted:
        score = 0
        total = len(expected_vals)
        st.markdown("**Result**")
        for idx, (picked, expected) in enumerate(zip(user_picks, expected_vals), start=1):
            if _is_blank_correct(picked, expected):
                score += 1
                st.success(f"Blank {idx}: Correct")
            else:
                st.error(f"Blank {idx}: Try again")
            st.caption(f"Your pick: {picked if picked != '(choose)' else '(empty)'}")
            st.caption(f"Expected: {_expected_to_text(expected)}")
        st.markdown(f"**Score: {score}/{total}**")
        st.progress(score / total if total else 0)
        if score == total and total > 0:
            st.balloons()


def render_stroke_canvas(char: str):
    safe_char = json.dumps(char)
    canvas_html = f"""
    <div style="padding:8px 0 2px 0;">
      <div style="font-weight:700;margin-bottom:8px;">Drawing Pad (write stroke by stroke)</div>
      <canvas id="hanzi-canvas" width="520" height="520" style="border:2px solid #cbd5e1;border-radius:10px;touch-action:none;background:#ffffff;"></canvas>
      <div style="margin-top:10px;display:flex;gap:8px;align-items:center;flex-wrap:wrap;">
        <button id="clear-btn" style="padding:6px 12px;border:1px solid #cbd5e1;border-radius:8px;background:#f8fafc;cursor:pointer;">Clear</button>
        <span style="font-size:13px;color:#374151;">Tip: follow the stroke numbers above, then draw each stroke in sequence.</span>
      </div>
    </div>
    <script>
      const canvas = document.getElementById("hanzi-canvas");
      const ctx = canvas.getContext("2d");
      const charToPractice = {safe_char};

      function drawGuide() {{
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        ctx.strokeStyle = "#e5e7eb";
        ctx.lineWidth = 1;

        const w = canvas.width;
        const h = canvas.height;
        const midX = w / 2;
        const midY = h / 2;

        ctx.beginPath();
        ctx.moveTo(midX, 0); ctx.lineTo(midX, h);
        ctx.moveTo(0, midY); ctx.lineTo(w, midY);
        ctx.moveTo(0, 0); ctx.lineTo(w, h);
        ctx.moveTo(w, 0); ctx.lineTo(0, h);
        ctx.stroke();

        ctx.fillStyle = "rgba(148, 163, 184, 0.22)";
        ctx.font = "260px 'Noto Sans SC', 'PingFang SC', 'Microsoft YaHei', sans-serif";
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText(charToPractice, midX, midY + 8);
      }}

      drawGuide();

      let drawing = false;

      function getPos(evt) {{
        const rect = canvas.getBoundingClientRect();
        const clientX = evt.touches ? evt.touches[0].clientX : evt.clientX;
        const clientY = evt.touches ? evt.touches[0].clientY : evt.clientY;
        return {{
          x: clientX - rect.left,
          y: clientY - rect.top
        }};
      }}

      function startDraw(evt) {{
        evt.preventDefault();
        drawing = true;
        const p = getPos(evt);
        ctx.beginPath();
        ctx.moveTo(p.x, p.y);
      }}

      function draw(evt) {{
        if (!drawing) return;
        evt.preventDefault();
        const p = getPos(evt);
        ctx.strokeStyle = "#111827";
        ctx.lineWidth = 8;
        ctx.lineCap = "round";
        ctx.lineJoin = "round";
        ctx.lineTo(p.x, p.y);
        ctx.stroke();
      }}

      function stopDraw(evt) {{
        if (!drawing) return;
        evt.preventDefault();
        drawing = false;
        ctx.closePath();
      }}

      canvas.addEventListener("mousedown", startDraw);
      canvas.addEventListener("mousemove", draw);
      canvas.addEventListener("mouseup", stopDraw);
      canvas.addEventListener("mouseleave", stopDraw);

      canvas.addEventListener("touchstart", startDraw, {{ passive: false }});
      canvas.addEventListener("touchmove", draw, {{ passive: false }});
      canvas.addEventListener("touchend", stopDraw, {{ passive: false }});

      document.getElementById("clear-btn").addEventListener("click", (e) => {{
        e.preventDefault();
        drawGuide();
      }});
    </script>
    """
    components.html(canvas_html, height=620, scrolling=False)


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

tab_vocab, tab_stroke, tab_convo, tab_quiz = st.tabs(
    ["Vocabulary Explorer", "Stroke Order Practice", "Daily Conversation", "Quick Quiz"]
)

with tab_vocab:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Filtered Words", f"{len(filtered):,}")
    col2.metric("1-char", f"{(filtered['character_count'] == 1).sum():,}")
    col3.metric("2-char", f"{(filtered['character_count'] == 2).sum():,}")
    col4.metric("3+ char", f"{(filtered['character_count'] >= 3).sum():,}")

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
    st.markdown("### Stroke Order Writing Exercise (with Number Sequence)")
    st.caption("Pick one Hanzi, follow the stroke numbers, and tick each stroke as you practice writing.")

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
            render_stroke_canvas(target_char)
        st.link_button("Open stroke source (SVG)", stroke_data["source_url"])

        st.markdown("#### Practice Checklist")
        checklist_cols = st.columns(4)
        for i, n in enumerate(stroke_data["sequence"]):
            with checklist_cols[i % 4]:
                st.checkbox(f"Stroke {n}", key=f"stroke_{target_char}_{n}")

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
    convo_ref_tab, convo_fill_tab = st.tabs(["Reference (Hanzi + Pinyin + English)", "Fill in the Blank"])

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
        render_fill_blank_activity(
            activity_id="daily_work_fill",
            title="Work Dialogue Practice",
            prompt="Fill the blanks from the word bank to rebuild the conversation.",
            word_bank=["有空", "怎么了", "问", "怎么样", "可能", "差不多了", "一起", "当然"],
            sentences=[
                {"text": "你现在 ___ 吗？", "answers": ["有空"]},
                {"text": "有，___？", "answers": ["怎么了"]},
                {"text": "我想 ___ 一下这个计划 ___。", "answers": ["问", "怎么样"]},
                {"text": "___ 要改一点，不过 ___。", "answers": ["可能", "差不多了"]},
            ],
        )
        st.divider()
        render_fill_blank_activity(
            activity_id="daily_commute_fill",
            title="Commute Dialogue Practice",
            prompt="Complete each line with the best phrase.",
            word_bank=["怎么", "地铁", "有时候", "公共汽车", "一起", "可以", "地铁站", "八点", "公司"],
            sentences=[
                {"text": "你 ___ 去公司？", "answers": ["怎么"]},
                {"text": "我坐 ___，___ 坐 ___。", "answers": ["地铁", "有时候", "公共汽车"]},
                {"text": "明天 ___ 去吗？", "answers": ["一起"]},
                {"text": "___，___ ___ 见。", "answers": ["可以", "地铁站", "八点"]},
            ],
        )
        st.divider()
        render_fill_blank_activity(
            activity_id="daily_food_fill",
            title="Food + Social Dialogue Practice",
            prompt="Use the word bank to complete the lines naturally.",
            word_bank=["吃饭", "还没有", "来不及", "早饭", "先", "餐厅", "不客气", "请"],
            sentences=[
                {"text": "你 ___ 了吗？", "answers": ["吃饭"]},
                {"text": "___，___ 吃 ___。", "answers": ["还没有", "来不及", "早饭"]},
                {"text": "那我们 ___ 去 ___ 吧。", "answers": ["先", "餐厅"]},
                {"text": "好啊，___，我 ___ 你。", "answers": ["不客气", "请"]},
            ],
        )

with tab_quiz:
    st.markdown("### Quick Quiz: Fill in the Blank Challenge")
    st.caption("Style: sentence blanks + word bank, like worksheet practice.")

    render_fill_blank_activity(
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

    render_fill_blank_activity(
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

st.info(
    "Tip: Keep official HSK filter broad, then use Level 1-10 to progressively increase difficulty from high-frequency daily words to advanced low-frequency words."
)
