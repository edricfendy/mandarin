import re
import unicodedata
from pathlib import Path

import pandas as pd
import streamlit as st

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

    max_rows = st.slider("Rows to show", min_value=20, max_value=500, value=120, step=20)

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

tab_vocab, tab_convo, tab_quiz = st.tabs(["Vocabulary Explorer", "Daily Conversation", "Quick Quiz"])

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

    shown = filtered[display_cols].head(max_rows).rename(
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

with tab_convo:
    st.markdown("### Daily Conversation (Hanzi + Pinyin + English)")
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

with tab_quiz:
    st.markdown("### Quick Quiz: Fill, Choose, and Score")
    st.caption("Write your own answers, then check instantly. You get score + per-question feedback.")

    quiz_items = [
        {
            "id": "q1",
            "type": "text_zh",
            "prompt": 'Translate to Mandarin: "I am at the subway station."',
            "answers": ["我在地铁站", "我在地铁站。"],
            "explain": "Use 在 for location: 我在 + place.",
        },
        {
            "id": "q2",
            "type": "text_en",
            "prompt": 'Translate to English: "我们以后再聊。"',
            "answers": ["lets talk later", "let us talk later", "we can talk later"],
            "explain": "以后再聊 = talk later / chat later.",
        },
        {
            "id": "q3",
            "type": "choice",
            "prompt": "Fill in blank: 我___你打电话。",
            "options": ["给", "在"],
            "answer": "给",
            "explain": "给 + person + 打电话 = call someone.",
        },
        {
            "id": "q4",
            "type": "choice",
            "prompt": "Fill in blank: 现在出发还___。",
            "options": ["来得及", "来不及"],
            "answer": "来得及",
            "explain": "来得及 = still enough time.",
        },
        {
            "id": "q5",
            "type": "choice",
            "prompt": 'Best reply to "谢谢你" is:',
            "options": ["不客气", "为什么"],
            "answer": "不客气",
            "explain": "不客气 means \"you're welcome.\"",
        },
        {
            "id": "q6",
            "type": "text_zh",
            "prompt": 'Translate to Mandarin: "Let\'s go together tomorrow."',
            "answers": ["我们明天一起去吧", "明天我们一起去吧"],
            "explain": "一起 + 去 + 吧 makes a natural suggestion.",
        },
        {
            "id": "q7",
            "type": "text_zh",
            "prompt": 'Fill in Hanzi: "kěnéng" (means maybe/possibly) = ___',
            "answers": ["可能"],
            "explain": "可能 = maybe / possible.",
        },
        {
            "id": "q8",
            "type": "text_zh",
            "prompt": 'Complete sentence in Hanzi: 地铁___就在前面。(station)',
            "answers": ["站", "地铁站"],
            "explain": "地铁站 = subway station.",
        },
    ]

    with st.form("interactive_quiz_form"):
        user_answers = {}
        for idx, item in enumerate(quiz_items, start=1):
            st.markdown(f"**{idx}. {item['prompt']}**")
            if item["type"] == "choice":
                user_answers[item["id"]] = st.radio(
                    "Choose one",
                    options=item["options"],
                    key=f"quiz_{item['id']}",
                    horizontal=True,
                )
            else:
                user_answers[item["id"]] = st.text_input("Your answer", key=f"quiz_{item['id']}")
        submitted = st.form_submit_button("Check My Answers")

    if submitted:
        score = 0
        st.markdown("### Result")
        for idx, item in enumerate(quiz_items, start=1):
            user_val = user_answers.get(item["id"], "")
            ok = False

            if item["type"] == "choice":
                ok = user_val == item["answer"]
                correct_text = item["answer"]
            elif item["type"] == "text_en":
                normalized = normalize_en_answer(user_val)
                ok = normalized in {normalize_en_answer(ans) for ans in item["answers"]}
                correct_text = item["answers"][0]
            else:
                normalized = normalize_zh_answer(user_val)
                ok = normalized in {normalize_zh_answer(ans) for ans in item["answers"]}
                correct_text = item["answers"][0]

            if ok:
                score += 1
                st.success(f"{idx}. Correct")
            else:
                st.error(f"{idx}. Not yet")

            st.caption(f"Your answer: {user_val if user_val else '(empty)'}")
            st.caption(f"Suggested answer: {correct_text}")
            st.caption(f"Tip: {item['explain']}")

        total = len(quiz_items)
        st.markdown(f"**Score: {score}/{total}**")
        st.progress(score / total)
        if score == total:
            st.balloons()

st.info(
    "Tip: Keep official HSK filter broad, then use Level 1-10 to progressively increase difficulty from high-frequency daily words to advanced low-frequency words."
)
