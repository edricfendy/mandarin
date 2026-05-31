const PROVIDERS = {
  openai: {
    keyEnv: "OPENAI_API_KEY",
    modelEnv: "OPENAI_MODEL",
    defaultModel: "gpt-5.4-mini",
    url: "https://api.openai.com/v1/responses",
    mode: "responses",
  },
  openrouter: {
    keyEnv: "OPENROUTER_API_KEY",
    modelEnv: "OPENROUTER_MODEL",
    defaultModel: "openai/gpt-oss-20b:free",
    url: "https://openrouter.ai/api/v1/chat/completions",
    mode: "chat",
  },
  ollama: {
    keyEnv: null,
    modelEnv: "OLLAMA_MODEL",
    defaultModel: "gpt-oss:20b",
    url: "http://127.0.0.1:11434/v1/chat/completions",
    urlEnv: "OLLAMA_BASE_URL",
    mode: "chat",
  },
  groq: {
    keyEnv: "GROQ_API_KEY",
    modelEnv: "GROQ_MODEL",
    defaultModel: "llama-3.3-70b-versatile",
    url: "https://api.groq.com/openai/v1/chat/completions",
    mode: "chat",
  },
  huggingface: {
    keyEnv: "HF_TOKEN",
    modelEnv: "HF_MODEL",
    defaultModel: "deepseek-ai/DeepSeek-R1:fastest",
    url: "https://router.huggingface.co/v1/chat/completions",
    mode: "chat",
  },
};

const PROVIDER_PRIORITY = ["openrouter", "groq", "huggingface", "openai"];

function sendJson(res, status, payload) {
  res.statusCode = status;
  res.setHeader("Content-Type", "application/json; charset=utf-8");
  res.end(JSON.stringify(payload));
}

function parseBody(req) {
  if (!req.body) return {};
  if (typeof req.body === "object") return req.body;
  try {
    return JSON.parse(req.body);
  } catch {
    return {};
  }
}

function sanitizeMessages(messages) {
  if (!Array.isArray(messages)) return [];
  return messages
    .filter((message) => ["user", "assistant"].includes(message?.role) && typeof message.content === "string")
    .slice(-12)
    .map((message) => ({
      role: message.role,
      content: message.content.slice(0, 1200),
    }));
}

function buildInstructions(level) {
  return [
    "You are a Mandarin Chinese conversation tutor.",
    `The learner level is ${level}.`,
    "Every reply must begin with exactly three rows.",
    "Row 1: a natural Mandarin Chinese reply in simplified Chinese.",
    "Row 2: Hanyu Pinyin with tone marks for row 1.",
    "Row 3: a concise English translation of row 1.",
    "Do not label the first three rows.",
    "If the learner makes a grammar, word choice, tone, or naturalness mistake, correct it gently.",
    "If there is a mistake, add one blank line after row 3, then a short Feedback section with Correction, Why, Better reply.",
    "If the learner is already correct, say it is natural and continue the conversation with one question.",
    "Keep the Mandarin row to one or two short sentences so the pinyin and English rows stay readable.",
  ].join(" ");
}

function resolveProvider() {
  const requested = String(process.env.AI_PROVIDER || "").toLowerCase().trim();
  if (requested && PROVIDERS[requested]) return { name: requested, config: PROVIDERS[requested] };

  const available = PROVIDER_PRIORITY.find((name) => process.env[PROVIDERS[name].keyEnv]);
  if (available) return { name: available, config: PROVIDERS[available] };

  return { name: requested || "openai", config: PROVIDERS.openai };
}

function providerModel(name, config) {
  return process.env.AI_MODEL || process.env[config.modelEnv] || config.defaultModel;
}

function providerKey(config) {
  return config.keyEnv ? process.env[config.keyEnv] : "";
}

function providerUrl(config) {
  const configuredUrl = config.urlEnv ? process.env[config.urlEnv] : "";
  if (!configuredUrl) return config.url;

  const trimmed = configuredUrl.replace(/\/+$/, "");
  if (trimmed.endsWith("/chat/completions")) return trimmed;
  if (trimmed.endsWith("/v1")) return `${trimmed}/chat/completions`;
  return `${trimmed}/v1/chat/completions`;
}

function missingKeyError(name, config) {
  const options = Object.entries(PROVIDERS)
    .map(([providerName, provider]) => `${providerName}: ${provider.keyEnv || "no API key"}`)
    .join(", ");
  return `AI tutor needs an API key. Current provider "${name}" expects ${config.keyEnv}. You can also set AI_PROVIDER plus one of: ${options}.`;
}

function extractResponsesText(payload) {
  if (typeof payload.output_text === "string") return payload.output_text.trim();

  const parts = [];
  for (const item of payload.output || []) {
    for (const content of item.content || []) {
      if (content.type === "output_text" && content.text) parts.push(content.text);
    }
  }
  return parts.join("\n").trim();
}

function extractChatText(payload) {
  return String(payload?.choices?.[0]?.message?.content || "").trim();
}

function buildChatRequest(providerName, config, apiKey, model, instructions, messages) {
  const headers = {
    "Content-Type": "application/json",
  };

  if (apiKey) headers.Authorization = `Bearer ${apiKey}`;

  if (providerName === "openrouter") {
    headers["HTTP-Referer"] = "https://mandarin.vercel.app";
    headers["X-OpenRouter-Title"] = "Mandarin Proficiency Trainer";
  }

  return {
    url: providerUrl(config),
    headers,
    body: {
      model,
      messages: [{ role: "system", content: instructions }, ...messages],
      temperature: 0.4,
      max_tokens: 700,
    },
  };
}

function buildResponsesRequest(config, apiKey, model, instructions, messages) {
  return {
    url: providerUrl(config),
    headers: {
      Authorization: `Bearer ${apiKey}`,
      "Content-Type": "application/json",
    },
    body: {
      model,
      instructions,
      input: messages,
      max_output_tokens: 700,
    },
  };
}

module.exports = async function handler(req, res) {
  if (req.method !== "POST") {
    res.setHeader("Allow", "POST");
    return sendJson(res, 405, { error: "Method not allowed." });
  }

  const { name: providerName, config } = resolveProvider();
  const apiKey = providerKey(config);
  if (config.keyEnv && !apiKey) return sendJson(res, 501, { error: missingKeyError(providerName, config) });

  const body = parseBody(req);
  const level = ["beginner", "intermediate", "advanced"].includes(body.level) ? body.level : "intermediate";
  const messages = sanitizeMessages(body.messages);
  const latest = messages[messages.length - 1];

  if (!latest || latest.role !== "user" || !latest.content.trim()) {
    return sendJson(res, 400, { error: "Send a Mandarin practice message first." });
  }

  const instructions = buildInstructions(level);
  const model = providerModel(providerName, config);
  const request =
    config.mode === "responses"
      ? buildResponsesRequest(config, apiKey, model, instructions, messages)
      : buildChatRequest(providerName, config, apiKey, model, instructions, messages);

  try {
    const modelResponse = await fetch(request.url, {
      method: "POST",
      headers: request.headers,
      body: JSON.stringify(request.body),
    });

    const data = await modelResponse.json().catch(() => ({}));
    if (!modelResponse.ok) {
      const detail = data?.error?.message || `${providerName} request failed.`;
      return sendJson(res, 502, { error: detail });
    }

    const reply = config.mode === "responses" ? extractResponsesText(data) : extractChatText(data);
    if (!reply) return sendJson(res, 502, { error: "The AI tutor returned an empty response." });
    return sendJson(res, 200, { reply, provider: providerName, model });
  } catch (error) {
    return sendJson(res, 502, { error: error.message || "The AI tutor could not respond." });
  }
};
