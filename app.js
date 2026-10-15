const DATA_URL = "data/vocab.json";
const PAGE_SIZE = 100;

const state = {
  vocab: [],
  filtered: [],
  page: 1,
  shuffleSeed: 0,
  selectedLevels: new Set(Array.from({ length: 10 }, (_, i) => String(i + 1))),
  selectedHsk: new Set(["1", "2", "3", "4", "5", "6", "7", "na"]),
  selectedLengths: new Set(["1", "2", "3", "4+"]),
  cueIndex: 0,
  cueFlipped: false,
  quizItems: [],
  aiMessages: [
    {
      role: "assistant",
      content:
        "你好！我们用中文聊天。你写一句中文，我会自然地回答，也会帮你改正语法、用词和语气。\nNǐ hǎo! Wǒmen yòng Zhōngwén liáotiān. Nǐ xiě yí jù Zhōngwén, wǒ huì zìrán de huídá, yě huì bāng nǐ gǎizhèng yǔfǎ, yòngcí hé yǔqì.\nHello! We will chat in Chinese. Write one Chinese sentence, and I will answer naturally and help correct grammar, word choice, and tone.",
    },
  ],
  aiBusy: false,
  aiVoice: {
    recognition: null,
    listening: false,
    supported: false,
    baseText: "",
    finalText: "",
    interimText: "",
    hadError: false,
    stopRequested: false,
    restartTimer: null,
  },
};

const $ = (id) => document.getElementById(id);

const PRACTICE_STATS_KEY = "mandarin-quest-stats-v1";
const DEFAULT_PRACTICE_STATS = {
  xp: 0,
  answered: 0,
  correct: 0,
  streak: 0,
  bestStreak: 0,
  lastDay: "",
  sessions: 0,
  completed: 0,
  seenWords: [],
  missedWords: [],
};

function loadPracticeStats() {
  try {
    const saved = JSON.parse(localStorage.getItem(PRACTICE_STATS_KEY) || "{}");
    const stats = Object.assign({}, DEFAULT_PRACTICE_STATS, saved);
    stats.seenWords = Array.isArray(stats.seenWords) ? stats.seenWords.slice(-1500) : [];
    stats.missedWords = Array.isArray(stats.missedWords) ? stats.missedWords.slice(-300) : [];
    return stats;
  } catch {
    return Object.assign({}, DEFAULT_PRACTICE_STATS);
  }
}

function savePracticeStats() {
  try {
    state.practiceStats.seenWords = state.practiceStats.seenWords.slice(-1500);
    state.practiceStats.missedWords = state.practiceStats.missedWords.slice(-300);
    localStorage.setItem(PRACTICE_STATS_KEY, JSON.stringify(state.practiceStats));
  } catch {
    // Practice still works when browser storage is unavailable.
  }
}

state.practiceStats = loadPracticeStats();
state.practiceDeck = [];
state.practiceSource = [];
state.practiceIndex = 0;
state.practiceQuestion = null;
state.practiceAnswered = false;
state.practiceFinished = false;
state.practiceMode = "mixed";
state.practiceScore = 0;
state.practiceSessionAnswered = 0;
state.practiceXpStart = 0;
state.practiceCombo = 0;
state.practiceLastType = "";

function localDayString(date) {
  const y = date.getFullYear();
  const m = String(date.getMonth() + 1).padStart(2, "0");
  const d = String(date.getDate()).padStart(2, "0");
  return y + "-" + m + "-" + d;
}

function shiftLocalDay(day, amount) {
  if (!day) return "";
  const date = new Date(day + "T12:00:00");
  date.setDate(date.getDate() + amount);
  return localDayString(date);
}

function updatePracticeStreak() {
  const stats = state.practiceStats;
  const today = localDayString(new Date());
  if (stats.lastDay === today) return;
  stats.streak = stats.lastDay === shiftLocalDay(today, -1) ? stats.streak + 1 : 1;
  stats.bestStreak = Math.max(stats.bestStreak || 0, stats.streak);
  stats.lastDay = today;
}

function renderPracticeStats() {
  const stats = state.practiceStats;
  const accuracy = stats.answered ? Math.round((stats.correct / stats.answered) * 100) : 0;
  const cards = [
    ["🔥", String(stats.streak || 0), "day streak"],
    ["✨", (stats.xp || 0).toLocaleString(), "total XP"],
    ["📚", (stats.seenWords || []).length.toLocaleString(), "words explored"],
    ["🎯", accuracy + "%", "overall accuracy"],
  ];
  $("practice-stats").innerHTML = cards.map(function (card) {
    return '<div class="quest-stat"><span class="quest-stat-icon">' + card[0] +
      '</span><div><strong>' + escapeHtml(card[1]) + '</strong><small>' +
      escapeHtml(card[2]) + '</small></div></div>';
  }).join("");
}

function shuffleList(items) {
  const result = items.slice();
  for (let i = result.length - 1; i > 0; i -= 1) {
    const j = Math.floor(Math.random() * (i + 1));
    const temp = result[i];
    result[i] = result[j];
    result[j] = temp;
  }
  return result;
}

function getPracticePool(level) {
  const seen = new Set();
  const all = state.vocab.filter(function (item) {
    if (!item || !item.h || !item.e || !String(item.e).trim()) return false;
    if (seen.has(item.h)) return false;
    seen.add(item.h);
    return true;
  });
  let pool = all;
  if (level === "beginner") {
    pool = all.filter(function (item) { return Number(item.v || 5) <= 3; });
  } else if (level === "intermediate") {
    pool = all.filter(function (item) {
      const value = Number(item.v || 5);
      return value >= 3 && value <= 7;
    });
  } else if (level === "advanced") {
    pool = all.filter(function (item) { return Number(item.v || 5) >= 6; });
  }
  return pool.length >= 20 ? pool : all;
}

function buildPracticeOptions(item, type, source) {
  const useMeaning = type === "meaning" || type === "listen";
  const correctLabel = useMeaning ? normalize(item.e) : item.h;
  const candidates = shuffleList(source.filter(function (candidate) {
    if (candidate.h === item.h) return false;
    const label = useMeaning ? normalize(candidate.e) : candidate.h;
    return label && label !== correctLabel;
  }));
  const usedLabels = new Set([correctLabel]);
  const options = [item];
  for (const candidate of candidates) {
    const label = useMeaning ? normalize(candidate.e) : candidate.h;
    if (!label || usedLabels.has(label)) continue;
    usedLabels.add(label);
    options.push(candidate);
    if (options.length === 4) break;
  }
  return shuffleList(options);
}

function choosePracticeType(item) {
  const selected = $("practice-mode").value;
  if (selected !== "mixed" && selected !== "review") {
    if (selected === "cloze" && !(item.xz && String(item.xz).includes(item.h))) return "meaning";
    return selected;
  }
  let types = ["meaning", "hanzi", "typing", "listen", "cloze"];
  if (!(item.xz && String(item.xz).includes(item.h))) types = types.filter(function (type) { return type !== "cloze"; });
  if (types.length > 1) types = types.filter(function (type) { return type !== state.practiceLastType; });
  return types[Math.floor(Math.random() * types.length)] || "meaning";
}

function startPracticeQuest() {
  if (!state.vocab.length) return;
  const level = $("practice-level").value;
  const count = Number($("practice-count").value || 10);
  const all = getPracticePool(level);
  const mode = $("practice-mode").value;
  let focusPool = all;

  if (mode === "review") {
    const missedSet = new Set(state.practiceStats.missedWords);
    const missed = all.filter(function (item) { return missedSet.has(item.h); });
    if (missed.length) focusPool = missed;
  }

  const seenSet = new Set(state.practiceStats.seenWords);
  const newWords = shuffleList(focusPool.filter(function (item) { return !seenSet.has(item.h); }));
  const familiarWords = shuffleList(focusPool.filter(function (item) { return seenSet.has(item.h); }));
  const orderedPool = newWords.concat(familiarWords);
  state.practiceDeck = orderedPool.slice(0, Math.min(count, orderedPool.length));
  if (!state.practiceDeck.length) state.practiceDeck = shuffleList(all).slice(0, count);
  state.practiceSource = all;
  state.practiceIndex = 0;
  state.practiceScore = 0;
  state.practiceSessionAnswered = 0;
  state.practiceXpStart = state.practiceStats.xp || 0;
  state.practiceCombo = 0;
  state.practiceLastType = "";
  state.practiceAnswered = false;
  state.practiceFinished = false;
  state.practiceMode = mode === "review" ? "mixed" : mode;
  state.practiceStats.sessions = (state.practiceStats.sessions || 0) + 1;
  savePracticeStats();
  renderPracticeStats();
  renderPracticeQuestion();
}

function practiceOptionMarkup(item, type) {
  const useMeaning = type === "meaning" || type === "listen";
  const label = useMeaning
    ? '<span class="practice-option-meaning">' + escapeHtml(item.e) + '</span>'
    : '<span class="practice-option-hanzi">' + escapeHtml(item.h) + '</span><small>' + escapeHtml(item.p || "") + '</small>';
  return '<button type="button" class="practice-option" data-practice-answer="' +
    escapeHtml(item.h) + '">' + label + '</button>';
}

function renderPracticeQuestion() {
  if (state.practiceIndex >= state.practiceDeck.length) {
    finishPracticeQuest();
    return;
  }

  const item = state.practiceDeck[state.practiceIndex];
  const type = choosePracticeType(item);
  state.practiceLastType = type;
  const question = {
    item: item,
    type: type,
    options: type === "typing" ? [] : buildPracticeOptions(item, type, state.practiceSource),
  };
  state.practiceQuestion = question;
  state.practiceAnswered = false;

  let label = "MEANING MATCH";
  let title = "What does this mean?";
  let prompt = '<div class="practice-prompt-hanzi">' + escapeHtml(item.h) + '</div><div class="practice-prompt-pinyin">' + escapeHtml(item.p || "") + '</div>';
  let audio = "";
  let typing = "";
  if (type === "hanzi") {
    label = "ENGLISH → MANDARIN";
    title = "How would you say this in Chinese?";
    prompt = '<div class="practice-prompt-english">' + escapeHtml(item.e) + '</div><div class="practice-prompt-note">Choose the matching characters.</div>';
  } else if (type === "typing") {
    label = "YOUR TURN TO TYPE";
    title = "Write the Mandarin word";
    prompt = '<div class="practice-prompt-english">' + escapeHtml(item.e) + '</div><div class="practice-prompt-note">Type Hanzi or pinyin with or without tone marks.</div>';
    typing = '<form id="practice-answer-form" class="practice-answer-form"><input id="practice-typed-answer" class="practice-typed-answer" type="text" autocomplete="off" placeholder="Type 中文 or pinyin…" aria-label="Your Mandarin answer" required><button type="submit" class="btn btn-primary">Check answer</button></form>';
  } else if (type === "listen") {
    label = "LISTEN & CHOOSE";
    title = "What did you hear?";
    prompt = '<div class="practice-audio-prompt"><span aria-hidden="true">🎧</span><span>Listen to the Mandarin, then choose its meaning.</span></div>';
    audio = '<button type="button" id="practice-play-audio" class="btn btn-secondary practice-audio-btn">🔊 Play Mandarin</button><span id="practice-audio-status" class="practice-audio-status"></span>';
  } else if (type === "cloze") {
    label = "USE THE CONTEXT";
    title = "Complete the example sentence";
    prompt = '<div class="practice-cloze-sentence">' + escapeHtml(String(item.xz).replace(item.h, "＿＿＿")) + '</div><div class="practice-prompt-note">Choose the word that fits the blank.</div>';
  }

  const progress = Math.round((state.practiceIndex / state.practiceDeck.length) * 100);
  const useMeaning = type === "meaning" || type === "listen";
  const options = question.options.map(function (option) {
    return practiceOptionMarkup(option, useMeaning ? "meaning" : type);
  }).join("");
  const modeName = {
    meaning: "Chinese → English",
    hanzi: "English → Chinese",
    typing: "Typing challenge",
    listen: "Listening challenge",
    cloze: "Sentence challenge",
  }[type] || "Practice";

  $("practice-game").innerHTML =
    '<div class="practice-progress-top"><span>QUESTION ' + (state.practiceIndex + 1) + ' OF ' + state.practiceDeck.length +
    '</span><span>🔥 ' + (state.practiceStats.streak || 0) + ' day streak</span></div>' +
    '<div class="practice-progress-track"><span style="width:' + progress + '%"></span></div>' +
    '<div class="practice-question-card">' +
      '<div class="practice-question-meta"><span class="practice-mode-tag">' + label + '</span><span class="practice-mode-name">' + modeName + '</span></div>' +
      '<h3>' + title + '</h3>' +
      '<div class="practice-prompt">' + prompt + '</div>' +
      audio +
      (type === "typing" ? typing : '<div class="practice-options">' + options + '</div>') +
      '<div id="practice-feedback" class="practice-feedback" hidden></div>' +
      '<div class="practice-question-footer"><span>One word at a time. You’re doing fine.</span><button type="button" id="practice-skip" class="practice-skip">Skip for now →</button></div>' +
    '</div>';

  $("practice-game").querySelectorAll("[data-practice-answer]").forEach(function (button) {
    button.addEventListener("click", function () { submitPracticeAnswer(button.dataset.practiceAnswer); });
  });
  const answerForm = $("practice-answer-form");
  if (answerForm) {
    answerForm.addEventListener("submit", function (event) {
      event.preventDefault();
      const input = $("practice-typed-answer");
      if (input && input.value.trim()) submitPracticeAnswer(input.value);
    });
  }
  const playButton = $("practice-play-audio");
  if (playButton) playButton.addEventListener("click", function () {
    const status = $("practice-audio-status");
    if (!("speechSynthesis" in window)) {
      status.textContent = "Audio is not available in this browser.";
      return;
    }
    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(item.h);
    utterance.lang = "zh-CN";
    utterance.rate = 0.78;
    window.speechSynthesis.speak(utterance);
    status.textContent = "Listen closely to the tones.";
  });
  $("practice-skip").addEventListener("click", function () {
    state.practiceIndex += 1;
    renderPracticeQuestion();
  });
}

function submitPracticeAnswer(answer) {
  if (!state.practiceQuestion || state.practiceAnswered) return;
  const item = state.practiceQuestion.item;
  const typed = normalize(answer);
  const correct = [item.h, item.p, item.pk].some(function (value) {
    return value && normalize(value) === typed;
  });
  completePracticeAnswer(correct, answer);
}

function completePracticeAnswer(correct, selectedAnswer) {
  if (state.practiceAnswered || !state.practiceQuestion) return;
  state.practiceAnswered = true;
  const question = state.practiceQuestion;
  const item = question.item;
  const stats = state.practiceStats;
  updatePracticeStreak();
  stats.answered += 1;
  state.practiceSessionAnswered += 1;
  let earned = 2;
  if (correct) {
    state.practiceScore += 1;
    stats.correct += 1;
    state.practiceCombo += 1;
    earned = 10 + Math.min(5, (state.practiceCombo - 1) * 2);
    stats.missedWords = stats.missedWords.filter(function (word) { return word !== item.h; });
  } else {
    state.practiceCombo = 0;
    if (!stats.missedWords.includes(item.h)) stats.missedWords.push(item.h);
  }
  stats.xp = (stats.xp || 0) + earned;
  if (!stats.seenWords.includes(item.h)) stats.seenWords.push(item.h);
  savePracticeStats();
  renderPracticeStats();

  const feedback = $("practice-feedback");
  const example = item.xz
    ? '<div class="practice-feedback-example"><strong>In context</strong><div class="practice-example-zh">' + escapeHtml(item.xz) +
      '</div><div class="practice-example-en">' + escapeHtml(item.xe || "") + '</div></div>'
    : "";
  feedback.hidden = false;
  feedback.className = "practice-feedback " + (correct ? "is-correct" : "is-incorrect");
  feedback.innerHTML =
    '<div class="practice-feedback-heading"><strong>' + (correct ? "太棒了！You got it!" : "Good effort — this is how it sticks.") +
    '</strong><span>+' + earned + ' XP</span></div>' +
    '<div class="practice-correct-answer"><span class="practice-answer-hanzi">' + escapeHtml(item.h) +
    '</span><span class="practice-answer-pinyin">' + escapeHtml(item.p || "") +
    '</span><span class="practice-answer-meaning">' + escapeHtml(item.e) + '</span></div>' +
    (correct ? '<p>Nice work. Keep the momentum going!</p>' : '<p>No stress—seeing the answer now helps you remember it next time.</p>') +
    example +
    '<button type="button" id="practice-next" class="btn btn-primary">' +
    (state.practiceIndex === state.practiceDeck.length - 1 ? "Finish quest ✓" : "Next question →") + '</button>';

  $("practice-game").querySelectorAll("[data-practice-answer]").forEach(function (button) {
    button.disabled = true;
    if (button.dataset.practiceAnswer === item.h) button.classList.add("is-correct");
    if (!correct && button.dataset.practiceAnswer === selectedAnswer) button.classList.add("is-wrong");
  });
  if (question.type === "typing") {
    const input = $("practice-typed-answer");
    if (input) input.disabled = true;
    const form = $("practice-answer-form");
    if (form) form.querySelector("button").disabled = true;
  } else {
  }
  $("practice-next").addEventListener("click", function () {
    state.practiceIndex += 1;
    renderPracticeQuestion();
  });
}

function finishPracticeQuest() {
  if (state.practiceFinished) return;
  state.practiceFinished = true;
  const stats = state.practiceStats;
  stats.completed = (stats.completed || 0) + 1;
  savePracticeStats();
  renderPracticeStats();
  const total = state.practiceDeck.length;
  const accuracy = state.practiceSessionAnswered
    ? Math.round((state.practiceScore / state.practiceSessionAnswered) * 100)
    : 0;
  const xpEarned = (stats.xp || 0) - state.practiceXpStart;
  let message = accuracy >= 80 ? "你做得很好！You’re building real momentum." :
    accuracy >= 50 ? "Good progress. A little review will make it stronger." :
    "Showing up is a win. Try the tricky words again when you’re ready.";
  $("practice-game").innerHTML =
    '<div class="practice-summary-card"><div class="practice-summary-emoji">' + (accuracy >= 80 ? "🏆" : accuracy >= 50 ? "🌟" : "🌱") +
    '</div><div class="practice-mode-tag">QUEST COMPLETE</div><h3>You showed up. That counts.</h3><p>' + message + '</p>' +
    '<div class="practice-summary-stats"><div><strong>' + state.practiceScore + '/' + total + '</strong><small>correct</small></div>' +
    '<div><strong>' + accuracy + '%</strong><small>accuracy</small></div>' +
    '<div><strong>+' + xpEarned + '</strong><small>XP earned</small></div></div>' +
    '<div class="practice-summary-actions"><button type="button" id="practice-again" class="btn btn-primary">Another round ↗</button>' +
    '<button type="button" id="practice-review" class="btn btn-secondary">Review tricky words 🔁</button></div></div>';
  $("practice-again").addEventListener("click", startPracticeQuest);
  $("practice-review").addEventListener("click", function () {
    $("practice-mode").value = "review";
    startPracticeQuest();
  });
}


function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function normalize(text) {
  return String(text ?? "")
    .toLowerCase()
    .normalize("NFD")
    .replace(/[\u0300-\u036f]/g, "")
    .replace(/[^a-z0-9\u4e00-\u9fff]+/g, " ")
    .trim();
}

function lengthKey(item) {
  if (item.c >= 4) return "4+";
  return String(item.c || 1);
}

function hskKey(item) {
  return item.l == null ? "na" : String(item.l);
}

function seededValue(text, seed) {
  let h = seed || 2166136261;
  for (let i = 0; i < text.length; i += 1) {
    h ^= text.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return h >>> 0;
}

function choice(items, seed) {
  if (!items.length) return null;
  return items[seededValue(String(seed), seed) % items.length];
}

function initChips(containerId, values, selectedSet, renderLabel) {
  const container = $(containerId);
  container.innerHTML = "";
  values.forEach((value) => {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = `chip ${selectedSet.has(String(value)) ? "active" : ""}`;
    btn.textContent = renderLabel(value);
    btn.addEventListener("click", () => {
      const key = String(value);
      if (selectedSet.has(key)) selectedSet.delete(key);
      else selectedSet.add(key);
      if (selectedSet.size === 0) selectedSet.add(key);
      btn.classList.toggle("active", selectedSet.has(key));
      applyFilters();
    });
    container.appendChild(btn);
  });
}

function applyFilters() {
  const keyword = normalize($("filter-keyword").value);
  const family = normalize($("filter-family").value);
  const sortMode = $("filter-sort").value;

  let rows = state.vocab.filter((item) => {
    if (!state.selectedLevels.has(String(item.v))) return false;
    if (!state.selectedHsk.has(hskKey(item))) return false;
    if (!state.selectedLengths.has(lengthKey(item))) return false;

    if (keyword) {
      const haystack = normalize(`${item.h} ${item.p} ${item.pk} ${item.e}`);
      if (!haystack.includes(keyword)) return false;
    }

    if (family) {
      const hanziHit = String(item.h).includes(family);
      const pinyinHit = normalize(`${item.p} ${item.pk}`).includes(family);
      if (!hanziHit && !pinyinHit) return false;
    }

    return true;
  });

  rows.sort((a, b) => {
    if (sortMode === "hanzi") return String(a.h).localeCompare(String(b.h), "zh-Hans-CN");
    if (sortMode === "frequency") return Number(a.f || 999999) - Number(b.f || 999999);
    if (sortMode === "proficiency") return Number(a.v || 0) - Number(b.v || 0) || Number(a.f || 999999) - Number(b.f || 999999);
    return String(a.pk || a.p).localeCompare(String(b.pk || b.p));
  });

  if (state.shuffleSeed) {
    rows = [...rows].sort((a, b) => seededValue(a.h, state.shuffleSeed) - seededValue(b.h, state.shuffleSeed));
  }

  state.filtered = rows;
  state.page = 1;
  renderAll();
}

function displayRows() {
  const showAll = $("filter-showall").checked;
  const maxRows = Math.max(20, Number($("filter-maxrows").value || 200));
  return showAll ? state.filtered : state.filtered.slice(0, maxRows);
}

function renderMetrics() {
  const rows = state.filtered;
  const metrics = [
    ["Filtered Words", rows.length.toLocaleString()],
    ["1-char", rows.filter((x) => x.c === 1).length.toLocaleString()],
    ["2-char", rows.filter((x) => x.c === 2).length.toLocaleString()],
    ["3+ char", rows.filter((x) => x.c >= 3).length.toLocaleString()],
  ];
  $("vocab-metrics").innerHTML = metrics
    .map(([label, value]) => `<div class="metric-card"><div class="metric-value">${value}</div><div class="metric-label">${label}</div></div>`)
    .join("");
}

function renderTable() {
  const rows = displayRows();
  const totalPages = Math.max(1, Math.ceil(rows.length / PAGE_SIZE));
  state.page = Math.min(state.page, totalPages);
  const start = (state.page - 1) * PAGE_SIZE;
  const pageRows = rows.slice(start, start + PAGE_SIZE);

  $("vocab-tbody").innerHTML = pageRows
    .map(
      (item) => `<tr>
        <td class="col-hanzi">${escapeHtml(item.h)}</td>
        <td class="col-pinyin">${escapeHtml(item.p)}</td>
        <td>${escapeHtml(item.e)}</td>
        <td>${escapeHtml(item.c)}</td>
        <td>${escapeHtml(item.v)}</td>
        <td>${item.l == null ? "N/A" : escapeHtml(item.l)}</td>
        <td>${escapeHtml(item.xz)}</td>
        <td>${escapeHtml(item.xe)}</td>
      </tr>`
    )
    .join("");

  const pages = [];
  const windowStart = Math.max(1, state.page - 3);
  const windowEnd = Math.min(totalPages, state.page + 3);
  for (let i = windowStart; i <= windowEnd; i += 1) pages.push(i);

  $("vocab-pagination").innerHTML = [
    `<button class="page-btn" data-page="${Math.max(1, state.page - 1)}">Prev</button>`,
    ...pages.map((p) => `<button class="page-btn ${p === state.page ? "active" : ""}" data-page="${p}">${p}</button>`),
    `<button class="page-btn" data-page="${Math.min(totalPages, state.page + 1)}">Next</button>`,
  ].join("");

  document.querySelectorAll(".page-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      state.page = Number(btn.dataset.page);
      renderTable();
    });
  });
}

function renderStroke() {
  const picker = $("stroke-picker");
  const singles = state.filtered.filter((item) => item.c === 1).slice(0, 700);
  const current = picker.value || singles[0]?.h || "你";
  picker.innerHTML = singles
    .map((item) => `<option value="${escapeHtml(item.h)}"${item.h === current ? " selected" : ""}>${escapeHtml(item.h)} - ${escapeHtml(item.p)} - ${escapeHtml(item.e).slice(0, 80)}</option>`)
    .join("");
  renderStrokeChar(extractFirstHanzi($("stroke-custom").value) || picker.value || "你");
}

function extractFirstHanzi(text) {
  const match = String(text || "").match(/[\u4e00-\u9fff]/);
  return match ? match[0] : "";
}

function renderStrokeChar(char) {
  const info = state.vocab.find((item) => item.h === char) || {};
  $("stroke-info").innerHTML = `<strong>${escapeHtml(char)}</strong> ${escapeHtml(info.p || "")} ${info.e ? `- ${escapeHtml(info.e)}` : ""}`;
  $("stroke-widget").innerHTML = `<div class="stroke-board-area">
    <div id="stroke-board-el"></div>
    <div class="stroke-btn-row">
      <button class="btn btn-secondary" id="stroke-animate">Animate</button>
      <button class="btn btn-secondary" id="stroke-quiz">Quiz</button>
      <button class="btn btn-secondary" id="stroke-reset">Reset</button>
    </div>
    <div class="stroke-status-text" id="stroke-status">Use the buttons to study stroke order.</div>
  </div>`;

  if (!window.HanziWriter) {
    $("stroke-status").textContent = "Stroke writer library did not load.";
    return;
  }

  const writer = HanziWriter.create("stroke-board-el", char, {
    width: 240,
    height: 240,
    padding: 10,
    showOutline: true,
    strokeAnimationSpeed: 1,
    delayBetweenStrokes: 180,
  });

  $("stroke-animate").addEventListener("click", () => writer.animateCharacter());
  $("stroke-quiz").addEventListener("click", () => {
    $("stroke-status").textContent = "Draw the character in the correct stroke order.";
    writer.quiz({
      onMistake: () => ($("stroke-status").textContent = "Close. Try that stroke again."),
      onCorrectStroke: () => ($("stroke-status").textContent = "Good stroke."),
      onComplete: () => ($("stroke-status").textContent = "Complete."),
    });
  });
  $("stroke-reset").addEventListener("click", () => {
    writer.showCharacter();
    $("stroke-status").textContent = "Reset.";
  });

  const related = state.vocab.filter((item) => String(item.h).includes(char)).slice(0, 18);
  $("stroke-related").innerHTML = `<h3>Related Words</h3><div class="dd-bank">${related
    .map((item) => `<span class="dd-token">${escapeHtml(item.h)} ${escapeHtml(item.p)}</span>`)
    .join("")}</div>`;
}

function renderConversations() {
  const conversations = [
    ["Work Conversation", [
      ["A", "你现在方便聊两分钟吗？", "Nǐ xiànzài fāngbiàn liáo liǎng fēnzhōng ma?", "Are you free to chat for two minutes right now?"],
      ["B", "可以，我刚开完会。", "Kěyǐ, wǒ gāng kāiwán huì.", "Sure, I just finished a meeting."],
      ["A", "客户希望我们把截止日期提前到周五。", "Kèhù xīwàng wǒmen bǎ jiézhǐ rìqī tíqián dào zhōuwǔ.", "The client wants us to move the deadline up to Friday."],
    ]],
    ["Commute Conversation", [
      ["A", "今天路上堵不堵？", "Jīntiān lùshang dǔ bu dǔ?", "Is traffic bad today?"],
      ["B", "高架有点慢，我改坐地铁了。", "Gāojià yǒudiǎn màn, wǒ gǎi zuò dìtiě le.", "The elevated road is slow, so I switched to the subway."],
      ["A", "那我们在公司楼下咖啡店见吧。", "Nà wǒmen zài gōngsī lóuxià kāfēidiàn jiàn ba.", "Then let's meet at the coffee shop under the office building."],
    ]],
  ];

  $("convo-ref").innerHTML = conversations
    .map(([title, lines]) => `<div class="convo-block"><h3 class="convo-block-title">${title}</h3>${lines
      .map(([role, zh, py, en]) => `<div class="dcx-wrap">
        <div class="dcx-speaker">${role}</div>
        <div class="dcx-token-hanzi">${escapeHtml(zh)}</div>
        <div class="dcx-token-pinyin">${escapeHtml(py)}</div>
        <div class="dcx-en">${escapeHtml(en)}</div>
      </div>`)
      .join("")}</div>`)
    .join("");

  renderDragPractice();
}

function renderDragPractice() {
  const rows = [
    ["你现在 ___ 吗？", "有空", "Are you free now?"],
    ["我想 ___ 一下这个计划。", "问", "I want to ask about this plan."],
    ["明天 ___ 去吗？", "一起", "Do you want to go together tomorrow?"],
    ["我们八点在 ___ 见。", "地铁站", "Let's meet at the subway station at eight."],
  ];
  const bank = rows.map((x) => x[1]).sort(() => 0.5 - Math.random());
  $("convo-drag").innerHTML = makeDragDrop("daily-drag", "Daily Practice", rows, bank);
  wireDragDrop("daily-drag", rows.map((x) => x[1]));
}

function makeDragDrop(id, title, rows, bank) {
  return `<div class="dd-wrap" id="${id}">
    <div class="dd-toolbar"><strong>${escapeHtml(title)}</strong><button class="btn btn-secondary dd-check">Check</button><span class="dd-tip">Drag each token into a blank.</span></div>
    <div class="dd-sentences">${rows
      .map(([text, answer, hint], idx) => `<div class="dd-row">
        <div class="dd-hint-english">${escapeHtml(hint)}</div>
        <div class="dd-line"><span class="dd-line-no">${idx + 1}.</span>${escapeHtml(text).replace("___", `<span class="dd-slot" data-answer="${escapeHtml(answer)}"></span>`)}</div>
      </div>`)
      .join("")}</div>
    <div class="dd-bank-label">Word Bank</div>
    <div class="dd-bank">${bank.map((word) => `<span class="dd-token" draggable="true">${escapeHtml(word)}</span>`).join("")}</div>
    <div class="dd-result"></div>
  </div>`;
}

function wireDragDrop(id) {
  const root = $(id);
  let dragged = null;
  root.querySelectorAll(".dd-token").forEach((token) => {
    token.addEventListener("dragstart", () => {
      dragged = token;
    });
  });
  root.querySelectorAll(".dd-slot").forEach((slot) => {
    slot.addEventListener("dragover", (event) => {
      event.preventDefault();
      slot.classList.add("over");
    });
    slot.addEventListener("dragleave", () => slot.classList.remove("over"));
    slot.addEventListener("drop", (event) => {
      event.preventDefault();
      slot.classList.remove("over", "correct", "wrong");
      if (dragged) slot.replaceChildren(dragged);
    });
  });
  root.querySelector(".dd-check").addEventListener("click", () => {
    let correct = 0;
    const slots = [...root.querySelectorAll(".dd-slot")];
    slots.forEach((slot) => {
      const text = slot.textContent.trim();
      const ok = text === slot.dataset.answer;
      slot.classList.toggle("correct", ok);
      slot.classList.toggle("wrong", !ok);
      if (ok) correct += 1;
    });
    root.querySelector(".dd-result").textContent = `${correct}/${slots.length} correct`;
  });
}

function renderQuiz() {
  const count = Number($("quiz-count").value || 24);
  $("quiz-count-display").textContent = count;
  const mode = document.querySelector("input[name='quiz-mode']:checked").value;
  const prompt = document.querySelector("input[name='quiz-prompt']:checked").value;
  const source = state.filtered.length ? state.filtered : state.vocab;
  const items = [...source].sort((a, b) => seededValue(a.h, Date.now()) - seededValue(b.h, Date.now())).slice(0, count);
  state.quizItems = items;

  if (mode === "translation") {
    $("quiz-area").innerHTML = `<div class="translation-form">${items
      .map((item, idx) => `<div class="tq-item">
        <div class="tq-prompt">${idx + 1}. ${escapeHtml(item.h)} ${prompt === "hanzi_pinyin" ? `<span class="col-pinyin">${escapeHtml(item.p)}</span>` : ""}</div>
        <div class="tq-hint">${escapeHtml(item.e)}</div>
        <input class="tq-input" data-answer="${escapeHtml(item.e)}" placeholder="Type the English meaning" />
        <div class="tq-result"></div>
      </div>`)
      .join("")}<button class="btn btn-primary" id="quiz-check">Check Answers</button><div class="tq-score" id="quiz-score"></div></div>`;
    $("quiz-check").addEventListener("click", checkTranslationQuiz);
    return;
  }

  const rows = items.slice(0, Math.min(count, 20)).map((item) => {
    const sentence = item.xz && String(item.xz).includes(item.h) ? String(item.xz).replace(item.h, "___") : `___ - ${item.e}`;
    return [sentence, item.h, item.e];
  });
  $("quiz-area").innerHTML = makeDragDrop("quiz-drag", "Dynamic Cloze Quiz", rows, rows.map((x) => x[1]).sort(() => 0.5 - Math.random()));
  wireDragDrop("quiz-drag");
}

function checkTranslationQuiz() {
  let correct = 0;
  const inputs = [...document.querySelectorAll(".tq-input")];
  inputs.forEach((input) => {
    const expected = normalize(input.dataset.answer);
    const actual = normalize(input.value);
    const ok = actual.length > 1 && expected.includes(actual);
    input.nextElementSibling.textContent = ok ? "Good" : `Expected: ${input.dataset.answer}`;
    input.nextElementSibling.className = `tq-result ${ok ? "good" : "bad"}`;
    if (ok) correct += 1;
  });
  $("quiz-score").textContent = `Score: ${correct}/${inputs.length}`;
}

function isSidebarOpen() {
  const sidebar = $("sidebar");
  if (sidebar.classList.contains("open")) return true;
  if (sidebar.classList.contains("closed")) return false;
  return window.matchMedia("(min-width: 861px)").matches;
}

function setSidebarOpen(open) {
  const sidebar = $("sidebar");
  sidebar.classList.toggle("open", open);
  sidebar.classList.toggle("closed", !open);
  $("sidebar-toggle").setAttribute("aria-expanded", String(open));
  $("sidebar-toggle").setAttribute("aria-label", open ? "Close filters" : "Open filters");
}

function setAiStatus(message, isError = false) {
  $("ai-status").textContent = message;
  $("ai-status").classList.toggle("error", isError);
}

function formatAiMessageContent(message) {
  if (message.role !== "assistant") return escapeHtml(message.content);

  const lines = String(message.content ?? "").split(/\r?\n/);
  let learningRow = 0;
  return lines
    .map((line) => {
      if (!line.trim() && learningRow < 3) return "";

      const rowClass =
        learningRow === 0
          ? "ai-row-zh"
          : learningRow === 1
            ? "ai-row-pinyin"
            : learningRow === 2
              ? "ai-row-en"
              : "ai-row-extra";
      if (line.trim() && learningRow < 3) learningRow += 1;
      return `<span class="ai-reply-row ${rowClass}">${escapeHtml(line) || "&nbsp;"}</span>`;
    })
    .join("");
}

function renderAiChat() {
  $("ai-chat-log").innerHTML = state.aiMessages
    .map((message) => {
      const roleLabel = message.role === "user" ? "You" : "AI";
      return `<div class="ai-message ${escapeHtml(message.role)}"><span class="ai-role">${roleLabel}</span>${formatAiMessageContent(message)}</div>`;
    })
    .join("");
  $("ai-chat-log").scrollTop = $("ai-chat-log").scrollHeight;
}

function pickMandarinVoice(lang) {
  if (!("speechSynthesis" in window)) return null;
  const voices = window.speechSynthesis.getVoices();
  return voices.find((voice) => voice.lang === lang) || voices.find((voice) => voice.lang.startsWith("zh")) || null;
}

function speakAiReply(text) {
  if (!$("ai-speak-replies").checked || !("speechSynthesis" in window)) return;
  window.speechSynthesis.cancel();
  const mandarinRow = String(text)
    .split(/\r?\n/)
    .map((line) => line.trim())
    .find(Boolean);
  const utterance = new SpeechSynthesisUtterance(mandarinRow || text);
  utterance.lang = $("ai-voice-lang").value;
  utterance.rate = 0.92;
  const voice = pickMandarinVoice(utterance.lang);
  if (voice) utterance.voice = voice;
  window.speechSynthesis.speak(utterance);
}

function updateVoiceButton() {
  const btn = $("ai-voice");
  btn.classList.toggle("listening", state.aiVoice.listening);
  btn.textContent = state.aiVoice.listening ? "Stop" : "Speak";
  btn.setAttribute("aria-pressed", String(state.aiVoice.listening));
}

function clearVoiceRestart() {
  if (!state.aiVoice.restartTimer) return false;
  clearTimeout(state.aiVoice.restartTimer);
  state.aiVoice.restartTimer = null;
  return true;
}

function currentVoiceText() {
  return [state.aiVoice.baseText, state.aiVoice.finalText, state.aiVoice.interimText].filter(Boolean).join(" ").trim();
}

function finishVoiceInput() {
  const spokenText = currentVoiceText();
  state.aiVoice.listening = false;
  state.aiVoice.stopRequested = false;
  state.aiVoice.interimText = "";
  updateVoiceButton();
  if (state.aiVoice.hadError) return;
  if (spokenText) sendAiMessage(null, spokenText);
  else if (!$("ai-status").classList.contains("error")) setAiStatus("No speech detected.", true);
}

function resetAiTutor() {
  if ("speechSynthesis" in window) window.speechSynthesis.cancel();
  if (state.aiVoice.listening && state.aiVoice.recognition) {
    const wasRestarting = clearVoiceRestart();
    state.aiVoice.stopRequested = true;
    state.aiVoice.hadError = true;
    if (wasRestarting) {
      state.aiVoice.listening = false;
      updateVoiceButton();
    } else {
      try {
        state.aiVoice.recognition.stop();
      } catch {
        state.aiVoice.listening = false;
        updateVoiceButton();
      }
    }
  }
  state.aiMessages = [
    {
      role: "assistant",
      content:
        "你好！我们重新开始。请用中文告诉我：你今天想练习什么话题？\nNǐ hǎo! Wǒmen chóngxīn kāishǐ. Qǐng yòng Zhōngwén gàosu wǒ: nǐ jīntiān xiǎng liànxí shénme huàtí?\nHello! Let us start again. Please tell me in Chinese: what topic do you want to practice today?",
    },
  ];
  setAiStatus("");
  renderAiChat();
}

function setAiBusy(isBusy) {
  state.aiBusy = isBusy;
  $("ai-send").disabled = isBusy;
  $("ai-chat-input").disabled = isBusy;
  $("ai-reset").disabled = isBusy;
  $("ai-voice").disabled = isBusy || !state.aiVoice.supported;
  if (isBusy) setAiStatus("Thinking...");
}

async function sendAiMessage(event, textOverride = "") {
  if (event) event.preventDefault();
  if (state.aiBusy) return;

  const input = $("ai-chat-input");
  const inputMode = textOverride ? "speech" : "typed";
  const text = (textOverride || input.value).trim();
  if (!text) {
    setAiStatus("Write a Mandarin sentence first.", true);
    return;
  }

  input.value = "";
  state.aiMessages.push({ role: "user", content: text });
  renderAiChat();
  setAiBusy(true);

  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        level: $("ai-level").value,
        topic: $("ai-topic").value,
        inputMode,
        messages: state.aiMessages.slice(-12),
      }),
    });
    const data = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(data.error || "The AI tutor could not respond.");
    state.aiMessages.push({ role: "assistant", content: data.reply });
    renderAiChat();
    setAiStatus("");
    speakAiReply(data.reply);
  } catch (error) {
    setAiStatus(error.message, true);
  } finally {
    setAiBusy(false);
  }
}

function setupVoiceTutor() {
  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  state.aiVoice.supported = Boolean(SpeechRecognition);
  if (!state.aiVoice.supported) {
    $("ai-voice").disabled = true;
    setAiStatus("Speech input is not supported in this browser.", true);
    return;
  }

  const recognition = new SpeechRecognition();
  recognition.continuous = true;
  recognition.interimResults = true;
  recognition.maxAlternatives = 1;
  state.aiVoice.recognition = recognition;

  recognition.onstart = () => {
    clearVoiceRestart();
    state.aiVoice.listening = true;
    updateVoiceButton();
    setAiStatus("Listening... press Stop when done.");
  };

  recognition.onresult = (event) => {
    let interim = "";
    for (let i = event.resultIndex; i < event.results.length; i += 1) {
      const transcript = event.results[i][0].transcript.trim();
      if (event.results[i].isFinal) state.aiVoice.finalText = `${state.aiVoice.finalText} ${transcript}`.trim();
      else interim = `${interim} ${transcript}`.trim();
    }
    state.aiVoice.interimText = interim;
    $("ai-chat-input").value = currentVoiceText();
  };

  recognition.onerror = (event) => {
    if (event.error === "no-speech" && !state.aiVoice.stopRequested) {
      setAiStatus("Still listening... press Stop when done.");
      return;
    }
    if (event.error === "aborted" && state.aiVoice.stopRequested) return;

    state.aiVoice.hadError = true;
    const message =
      event.error === "not-allowed"
        ? "Microphone permission was blocked."
        : event.error === "audio-capture"
          ? "No microphone was found."
          : "Voice input stopped. Please try again.";
    setAiStatus(message, true);
  };

  recognition.onend = () => {
    if (!state.aiVoice.stopRequested && !state.aiVoice.hadError) {
      state.aiVoice.listening = true;
      updateVoiceButton();
      setAiStatus("Still listening... press Stop when done.");
      clearVoiceRestart();
      state.aiVoice.restartTimer = setTimeout(() => {
        try {
          recognition.start();
        } catch {
          state.aiVoice.listening = false;
          updateVoiceButton();
          setAiStatus("Voice input paused. Press Speak to continue.", true);
        }
      }, 350);
      return;
    }

    clearVoiceRestart();
    finishVoiceInput();
  };

  $("ai-voice-lang").addEventListener("change", () => {
    recognition.lang = $("ai-voice-lang").value;
  });
  recognition.lang = $("ai-voice-lang").value;
  updateVoiceButton();
}

function toggleVoiceInput() {
  if (!state.aiVoice.supported || !state.aiVoice.recognition || state.aiBusy) return;
  if (state.aiVoice.listening) {
    const wasRestarting = clearVoiceRestart();
    state.aiVoice.stopRequested = true;
    if (wasRestarting) {
      finishVoiceInput();
      return;
    }
    try {
      state.aiVoice.recognition.stop();
    } catch {
      finishVoiceInput();
    }
    return;
  }
  if ("speechSynthesis" in window) window.speechSynthesis.cancel();
  clearVoiceRestart();
  state.aiVoice.baseText = $("ai-chat-input").value.trim();
  state.aiVoice.finalText = "";
  state.aiVoice.interimText = "";
  state.aiVoice.hadError = false;
  state.aiVoice.stopRequested = false;
  state.aiVoice.recognition.lang = $("ai-voice-lang").value;
  try {
    state.aiVoice.recognition.start();
  } catch {
    setAiStatus("Voice input is already starting.", true);
  }
}

function renderCueCards() {
  const deck = displayRows().length ? displayRows() : state.vocab;
  if (!deck.length) return;
  state.cueIndex = Math.max(0, Math.min(state.cueIndex, deck.length - 1));
  const item = deck[state.cueIndex];
  const front = `<div class="cue-face-front-hanzi">${escapeHtml(item.h)}</div>
    <div class="cue-face-front-pinyin">${escapeHtml(item.p)}</div>
    <div class="cue-face-front-meaning">${escapeHtml(item.e)}</div>`;
  const back = `<div class="cue-back-label">Usage</div>
    <div class="cue-back-zh">${escapeHtml(item.xz || item.h)}</div>
    <div class="cue-back-pinyin">${escapeHtml(item.p)}</div>
    <div class="cue-back-en">${escapeHtml(item.xe || item.e)}</div>`;

  $("cue-widget").innerHTML = `<div class="cue-wrap">
    <div class="cue-help">Click the card to flip.</div>
    <div class="cue-index">${state.cueIndex + 1} / ${deck.length}</div>
    <div class="cue-card" id="cue-card">
      <button class="cue-arrow cue-arrow-left" id="cue-prev" aria-label="Previous">‹</button>
      <div>${state.cueFlipped ? back : front}</div>
      <button class="cue-arrow cue-arrow-right" id="cue-next" aria-label="Next">›</button>
    </div>
  </div>`;

  $("cue-card").addEventListener("click", () => {
    state.cueFlipped = !state.cueFlipped;
    renderCueCards();
  });
  $("cue-prev").addEventListener("click", (event) => {
    event.stopPropagation();
    state.cueIndex = (state.cueIndex - 1 + deck.length) % deck.length;
    state.cueFlipped = false;
    renderCueCards();
  });
  $("cue-next").addEventListener("click", (event) => {
    event.stopPropagation();
    state.cueIndex = (state.cueIndex + 1) % deck.length;
    state.cueFlipped = false;
    renderCueCards();
  });
}

function renderAll() {
  renderMetrics();
  renderTable();
  renderStroke();
  renderConversations();
  renderCueCards();
}

function downloadCsv() {
  const header = ["hanzi", "pinyin", "english", "chars", "level", "hsk", "example_zh", "example_en"];
  const rows = state.filtered.map((item) => [item.h, item.p, item.e, item.c, item.v, item.l ?? "", item.xz, item.xe]);
  const csv = [header, ...rows]
    .map((row) => row.map((cell) => `"${String(cell ?? "").replaceAll('"', '""')}"`).join(","))
    .join("\n");
  const blob = new Blob([csv], { type: "text/csv;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = "mandarin_filtered_vocab.csv";
  link.click();
  URL.revokeObjectURL(url);
}

function wireEvents() {
  initChips("filter-proficiency", Array.from({ length: 10 }, (_, i) => i + 1), state.selectedLevels, (v) => `L${v}`);
  initChips("filter-hsk", ["1", "2", "3", "4", "5", "6", "7", "na"], state.selectedHsk, (v) => (v === "na" ? "N/A" : `HSK ${v}`));
  initChips("filter-length", ["1", "2", "3", "4+"], state.selectedLengths, (v) => (v === "4+" ? "4+ chars" : `${v} char${v === "1" ? "" : "s"}`));

  ["filter-keyword", "filter-family", "filter-sort", "filter-maxrows", "filter-showall"].forEach((id) => {
    $(id).addEventListener("input", applyFilters);
    $(id).addEventListener("change", applyFilters);
  });

  $("btn-shuffle-examples").addEventListener("click", () => {
    state.shuffleSeed = Date.now();
    applyFilters();
  });
  $("btn-download-csv").addEventListener("click", downloadCsv);
  $("stroke-picker").addEventListener("change", () => renderStrokeChar($("stroke-picker").value));
  $("stroke-custom").addEventListener("input", () => {
    const char = extractFirstHanzi($("stroke-custom").value);
    if (char) renderStrokeChar(char);
  });

  $("sidebar-toggle").setAttribute("aria-expanded", String(isSidebarOpen()));
  $("sidebar-toggle").addEventListener("click", () => setSidebarOpen(!isSidebarOpen()));
  $("sidebar-close").addEventListener("click", () => setSidebarOpen(false));
  window.addEventListener("resize", () => {
    $("sidebar-toggle").setAttribute("aria-expanded", String(isSidebarOpen()));
  });

  document.querySelectorAll(".tab-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".tab-btn").forEach((x) => x.classList.remove("active"));
      document.querySelectorAll(".tab-panel").forEach((x) => x.classList.remove("active"));
      btn.classList.add("active");
      $(`panel-${btn.dataset.tab}`).classList.add("active");
    });
  });

  document.querySelectorAll(".convo-subtab").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".convo-subtab").forEach((x) => x.classList.remove("active"));
      document.querySelectorAll(".convo-panel").forEach((x) => x.classList.remove("active"));
      btn.classList.add("active");
      $(`convo-${btn.dataset.convoTab}`).classList.add("active");
    });
  });

  $("practice-start").addEventListener("click", startPracticeQuest);
  $("practice-start-inline").addEventListener("click", startPracticeQuest);
  $("quiz-count").addEventListener("input", () => ($("quiz-count-display").textContent = $("quiz-count").value));
  $("btn-generate-quiz").addEventListener("click", renderQuiz);
  $("ai-chat-form").addEventListener("submit", sendAiMessage);
  $("ai-reset").addEventListener("click", resetAiTutor);
  $("ai-voice").addEventListener("click", toggleVoiceInput);
  setupVoiceTutor();
  renderAiChat();
}

async function init() {
  try {
    wireEvents();
    const response = await fetch(DATA_URL);
    if (!response.ok) throw new Error(`Could not load ${DATA_URL}`);
    state.vocab = await response.json();
    state.filtered = state.vocab;
    applyFilters();
    renderPracticeStats();
    renderQuiz();
    $("app-loader").classList.add("hidden");
    $("app").classList.remove("hidden");
  } catch (error) {
    $("app-loader").innerHTML = `<div class="loader-content"><div class="loader-icon">汉</div><div class="loader-text">${escapeHtml(error.message)}</div></div>`;
  }
}

document.addEventListener("DOMContentLoaded", init);
