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
      content: "你好！我们用中文聊天。你写一句中文，我会自然地回答，也会帮你改正语法、用词和语气。",
    },
  ],
  aiBusy: false,
};

const $ = (id) => document.getElementById(id);

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

function renderAiChat() {
  $("ai-chat-log").innerHTML = state.aiMessages
    .map((message) => {
      const roleLabel = message.role === "user" ? "You" : "AI";
      return `<div class="ai-message ${escapeHtml(message.role)}"><span class="ai-role">${roleLabel}</span>${escapeHtml(message.content)}</div>`;
    })
    .join("");
  $("ai-chat-log").scrollTop = $("ai-chat-log").scrollHeight;
}

function resetAiTutor() {
  state.aiMessages = [
    {
      role: "assistant",
      content: "你好！我们重新开始。请用中文告诉我：你今天想练习什么话题？",
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
  if (isBusy) setAiStatus("Thinking...");
}

async function sendAiMessage(event) {
  event.preventDefault();
  if (state.aiBusy) return;

  const input = $("ai-chat-input");
  const text = input.value.trim();
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
        messages: state.aiMessages.slice(-12),
      }),
    });
    const data = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(data.error || "The AI tutor could not respond.");
    state.aiMessages.push({ role: "assistant", content: data.reply });
    renderAiChat();
    setAiStatus("");
  } catch (error) {
    setAiStatus(error.message, true);
  } finally {
    setAiBusy(false);
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

  $("quiz-count").addEventListener("input", () => ($("quiz-count-display").textContent = $("quiz-count").value));
  $("btn-generate-quiz").addEventListener("click", renderQuiz);
  $("ai-chat-form").addEventListener("submit", sendAiMessage);
  $("ai-reset").addEventListener("click", resetAiTutor);
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
    renderQuiz();
    $("app-loader").classList.add("hidden");
    $("app").classList.remove("hidden");
  } catch (error) {
    $("app-loader").innerHTML = `<div class="loader-content"><div class="loader-icon">汉</div><div class="loader-text">${escapeHtml(error.message)}</div></div>`;
  }
}

document.addEventListener("DOMContentLoaded", init);
