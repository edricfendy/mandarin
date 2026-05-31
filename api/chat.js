const OPENAI_API_URL = "https://api.openai.com/v1/responses";
const DEFAULT_MODEL = "gpt-5.4-mini";

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

function extractOutputText(payload) {
  if (typeof payload.output_text === "string") return payload.output_text.trim();

  const parts = [];
  for (const item of payload.output || []) {
    for (const content of item.content || []) {
      if (content.type === "output_text" && content.text) parts.push(content.text);
    }
  }
  return parts.join("\n").trim();
}

module.exports = async function handler(req, res) {
  if (req.method !== "POST") {
    res.setHeader("Allow", "POST");
    return sendJson(res, 405, { error: "Method not allowed." });
  }

  const apiKey = process.env.OPENAI_API_KEY;
  if (!apiKey) {
    return sendJson(res, 501, {
      error: "AI tutor is deployed, but OPENAI_API_KEY is not configured in Vercel.",
    });
  }

  const body = parseBody(req);
  const level = ["beginner", "intermediate", "advanced"].includes(body.level) ? body.level : "intermediate";
  const messages = sanitizeMessages(body.messages);
  const latest = messages[messages.length - 1];

  if (!latest || latest.role !== "user" || !latest.content.trim()) {
    return sendJson(res, 400, { error: "Send a Mandarin practice message first." });
  }

  const instructions = [
    "You are a Mandarin Chinese conversation tutor.",
    `The learner level is ${level}.`,
    "Reply mainly in Mandarin Chinese, using short English only when needed for clarity.",
    "If the learner makes a grammar, word choice, tone, or naturalness mistake, correct it gently.",
    "Use this compact structure when there is a mistake: Correction, Why, Better reply.",
    "If the learner is already correct, say it is natural and continue the conversation with one question.",
    "Keep replies concise enough for a language practice chat.",
  ].join(" ");

  try {
    const openaiResponse = await fetch(OPENAI_API_URL, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${apiKey}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        model: process.env.OPENAI_MODEL || DEFAULT_MODEL,
        instructions,
        input: messages,
        max_output_tokens: 700,
      }),
    });

    const data = await openaiResponse.json().catch(() => ({}));
    if (!openaiResponse.ok) {
      const detail = data?.error?.message || "OpenAI request failed.";
      return sendJson(res, 502, { error: detail });
    }

    const reply = extractOutputText(data);
    if (!reply) return sendJson(res, 502, { error: "The AI tutor returned an empty response." });
    return sendJson(res, 200, { reply });
  } catch (error) {
    return sendJson(res, 502, { error: error.message || "The AI tutor could not respond." });
  }
};
