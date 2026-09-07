// Tests bun pour le proxy Anthropic→OpenAI (src/fast_proxy.ts).
// Lance : bun test tests/fast_proxy.test.ts

import { describe, test, expect, beforeAll, afterAll } from "bun:test";

const MOCK_PORT = 18082;
const PROXY_PORT = 18081;

let mockServer: any;
let handleRequest: (req: Request) => Promise<Response>;
let lastUpstreamBody: any = null;

beforeAll(async () => {
  // Mock llama-server : SSE si stream, JSON sinon.
  mockServer = Bun.serve({
    port: MOCK_PORT,
    async fetch(req) {
      const url = new URL(req.url);
      if (url.pathname === "/health") return new Response('{"status":"ok"}');
      if (req.method !== "POST" || url.pathname !== "/v1/chat/completions") {
        return new Response("not found", { status: 404 });
      }
      const body = await req.json();
      lastUpstreamBody = body;
      if (body.model === "qwen-upstream-error") {
        return Response.json(
          { error: { code: 500, message: "Jinja Exception: System message must be at the beginning.", type: "server_error" } },
          { status: 500 }
        );
      }
      const emptyContent = body.model === "qwen-reasoning-empty";
      const openaiResp = {
        id: "chatcmpl-test",
        choices: [{
          index: 0, finish_reason: "stop",
          message: emptyContent
            ? { role: "assistant", content: "", reasoning_content: "réflexion tronquée" }
            : { role: "assistant", content: "OK" },
        }],
        usage: { prompt_tokens: 5, completion_tokens: 1, total_tokens: 6 },
      };
      if (body.stream) {
        const sse =
          `data: ${JSON.stringify({ choices: [{ delta: { role: "assistant", content: null } }] })}\n\n` +
          `data: ${JSON.stringify({ choices: [{ delta: { content: "OK" } }] })}\n\n` +
          `data: ${JSON.stringify({ choices: [{ delta: {}, finish_reason: "stop" }] })}\n\n` +
          `data: [DONE]\n\n`;
        return new Response(sse, { headers: { "Content-Type": "text/event-stream" } });
      }
      return Response.json(openaiResp);
    },
  });

  const mod = await import("../src/fast_proxy.ts");
  handleRequest = mod.handleRequest;
});

afterAll(() => {
  mockServer?.stop(true);
});

function proxyReq(body: object): Request {
  return new Request(`http://localhost:${PROXY_PORT}/v1/messages`, {
    method: "POST",
    headers: { "Content-Type": "application/json", "x-api-key": "sk-local-proxy", "anthropic-version": "2023-06-01" },
    body: JSON.stringify(body),
  });
}

const MOCK_UPSTREAM = `http://localhost:${MOCK_PORT}/v1`;

async function callProxy(body: object): Promise<Response> {
  return handleRequest(proxyReq(body), MOCK_UPSTREAM);
}

describe("fast_proxy /v1/messages", () => {
  test("body sans champ stream → SSE en sortie (ne parse jamais le SSE comme JSON)", async () => {
    const res = await callProxy({ model: "qwen", max_tokens: 20, messages: [{ role: "user", content: "hi" }] });
    expect(res.status).toBe(200);
    expect(res.headers.get("content-type")).toContain("text/event-stream");
    const text = await res.text();
    expect(text).toContain("event: message_start");
    expect(text).toContain("text_delta");
    expect(text).toContain("OK");
    expect(text).toContain("event: message_stop");
  });

  test("stream:true → SSE en sortie", async () => {
    const res = await callProxy({ model: "qwen", max_tokens: 20, stream: true, messages: [{ role: "user", content: "hi" }] });
    expect(res.status).toBe(200);
    expect(res.headers.get("content-type")).toContain("text/event-stream");
    const text = await res.text();
    expect(text).toContain("event: message_start");
    expect(text).toContain("OK");
  });

  test("stream:false → JSON Anthropic (pas SSE)", async () => {
    const res = await callProxy({ model: "qwen", max_tokens: 20, stream: false, messages: [{ role: "user", content: "hi" }] });
    expect(res.status).toBe(200);
    expect(res.headers.get("content-type")).toContain("application/json");
    const json = await res.json();
    expect(json.type).toBe("message");
    expect(json.content[0].text).toBe("OK");
    expect(json.usage.total_tokens).toBe(6);
  });

  test("system Anthropic → message system OpenAI en amont", async () => {
    lastUpstreamBody = null;
    await callProxy({ model: "qwen", max_tokens: 20, stream: false, system: "tu es un test", messages: [{ role: "user", content: "hi" }] });
    expect(lastUpstreamBody).not.toBeNull();
    expect(lastUpstreamBody.messages[0]).toEqual({ role: "system", content: "tu es un test" });
  });

  test("alias claude-sonnet-4-6 → qwen3 en amont", async () => {
    const { mapModel } = await import("../src/fast_proxy.ts");
    expect(mapModel("claude-sonnet-4-6")).toBe("qwen3");
    expect(mapModel("qwen3")).toBe("qwen3");
  });

  test("REASONING_EFFORT env → reasoning_effort en amont, absent sinon", async () => {
    process.env.REASONING_EFFORT = "none";
    lastUpstreamBody = null;
    await callProxy({ model: "qwen", max_tokens: 20, stream: false, messages: [{ role: "user", content: "hi" }] });
    expect(lastUpstreamBody.reasoning_effort).toBe("none");
    delete process.env.REASONING_EFFORT;
    lastUpstreamBody = null;
    await callProxy({ model: "qwen", max_tokens: 20, stream: false, messages: [{ role: "user", content: "hi" }] });
    expect("reasoning_effort" in lastUpstreamBody).toBe(false);
  });

  test("health proxy → upstream health", async () => {
    const res = await handleRequest(new Request(`http://localhost:${PROXY_PORT}/health`), MOCK_UPSTREAM);
    expect(res.status).toBe(200);
    expect(await res.text()).toContain("ok");
  });

  test("bloc image Anthropic (base64) → image_url data-uri OpenAI en amont", async () => {
    lastUpstreamBody = null;
    await callProxy({
      model: "qwen",
      max_tokens: 20,
      stream: false,
      messages: [{
        role: "user",
        content: [
          { type: "text", text: "décris cette image" },
          { type: "image", source: { type: "base64", media_type: "image/png", data: "iVBORw0KGgo=" } },
        ],
      }],
    });
    expect(lastUpstreamBody).not.toBeNull();
    const parts = lastUpstreamBody.messages[0].content;
    expect(parts[0]).toEqual({ type: "text", text: "décris cette image" });
    expect(parts[1]).toEqual({
      type: "image_url",
      image_url: { url: "data:image/png;base64,iVBORw0KGgo=" },
    });
  });

  test("content texte simple (string) reste inchangé en amont", async () => {
    lastUpstreamBody = null;
    await callProxy({ model: "qwen", max_tokens: 20, stream: false, messages: [{ role: "user", content: "hi" }] });
    expect(lastUpstreamBody.messages[0].content).toBe("hi");
  });

  test("thinking désactivé par défaut (comportement actuel verrouillé)", async () => {
    lastUpstreamBody = null;
    await callProxy({ model: "qwen", max_tokens: 20, stream: false, messages: [{ role: "user", content: "hi" }] });
    expect(lastUpstreamBody.chat_template_kwargs.enable_thinking).toBe(false);
  });

  test("stream:false replie message.reasoning_content si content vide (budget consommé par le thinking)", async () => {
    const res = await callProxy({
      model: "qwen-reasoning-empty", max_tokens: 20, stream: false,
      messages: [{ role: "user", content: "hi" }],
    });
    const json = await res.json();
    expect(json.content[0].text).toBe("réflexion tronquée");
  });

  test("ENABLE_THINKING=1 → enable_thinking true en amont", async () => {
    process.env.ENABLE_THINKING = "1";
    lastUpstreamBody = null;
    await callProxy({ model: "qwen", max_tokens: 20, stream: false, messages: [{ role: "user", content: "hi" }] });
    expect(lastUpstreamBody.chat_template_kwargs.enable_thinking).toBe(true);
    delete process.env.ENABLE_THINKING;
  });

  test("message role=system en fin de tableau → fusionné dans le system prompt amont, jamais orphelin", async () => {
    lastUpstreamBody = null;
    await callProxy({
      model: "qwen", max_tokens: 20, stream: false,
      messages: [
        { role: "user", content: "hi" },
        { role: "system", content: "reminder contextuel" },
      ],
    });
    const roles = lastUpstreamBody.messages.map((m: any) => m.role);
    expect(roles.filter((r: string) => r === "system").length).toBeLessThanOrEqual(1);
    if (roles.includes("system")) {
      expect(roles[0]).toBe("system");
    }
  });

  test("upstream 500 en stream → erreur propagée, jamais un succès vide", async () => {
    const res = await callProxy({ model: "qwen-upstream-error", max_tokens: 20, stream: true, messages: [{ role: "user", content: "hi" }] });
    expect(res.status).not.toBe(200);
  });

  test("upstream 500 en non-stream → erreur propagée, pas de content vide silencieux", async () => {
    const res = await callProxy({ model: "qwen-upstream-error", max_tokens: 20, stream: false, messages: [{ role: "user", content: "hi" }] });
    expect(res.status).not.toBe(200);
  });
});