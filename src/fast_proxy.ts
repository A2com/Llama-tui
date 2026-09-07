const UPSTREAM = process.env.UPSTREAM || "http://localhost:8082/v1";
const PORT = parseInt(process.env.PORT || "8001", 10);

export function mapModel(name: string): string {
  const aliases: Record<string, string> = {
    "claude-sonnet-4-6": "qwen3",
    "claude-opus-4-7": "qwen3",
    "claude-haiku-4-5-20251001": "qwen3",
    "claude-3-5-sonnet-20241022": "qwen3",
  };
  return aliases[name] || name;
}

function convertContent(content: any): any {
  if (typeof content !== "string" && !Array.isArray(content)) return content;
  if (typeof content === "string") return content;
  return content.map((block: any) => {
    if (block.type === "image" && block.source?.type === "base64") {
      return {
        type: "image_url",
        image_url: { url: `data:${block.source.media_type};base64,${block.source.data}` },
      };
    }
    return block;
  });
}

export async function handleRequest(req: Request, upstreamBase: string = UPSTREAM): Promise<Response> {
  const url = new URL(req.url);

  if (url.pathname === "/health") {
    try {
      const upstream = await fetch(upstreamBase.replace("/v1", "") + "/health", { signal: AbortSignal.timeout(2000) });
      return new Response(upstream.body, { status: upstream.status, headers: upstream.headers });
    } catch {
      return new Response("upstream down", { status: 503 });
    }
  }

  if (url.pathname === "/v1/models") {
    const upstream = await fetch(upstreamBase + "/models", { headers: { Authorization: "Bearer fake" } });
    return new Response(upstream.body, { status: upstream.status, headers: upstream.headers });
  }

  if (req.method === "POST" && url.pathname === "/v1/messages") {
    const anthropicBody = await req.json();

    const systemParts: any[] = [];
    if (anthropicBody.system) {
      const converted = convertContent(anthropicBody.system);
      if (Array.isArray(converted)) systemParts.push(...converted);
      else systemParts.push(converted);
    }

    const openaiMessages: any[] = [];
    for (const msg of anthropicBody.messages) {
      if (msg.role === "system" || msg.role === "developer") {
        const converted = convertContent(msg.content);
        if (Array.isArray(converted)) systemParts.push(...converted);
        else systemParts.push(converted);
        continue;
      }
      openaiMessages.push({ role: msg.role, content: convertContent(msg.content) });
    }

    if (systemParts.length > 0) {
      const systemContent = systemParts.length === 1 && typeof systemParts[0] === "string"
        ? systemParts[0]
        : systemParts;
      openaiMessages.unshift({ role: "system", content: systemContent });
    }

    const wantStream = anthropicBody.stream ?? true;
    const openaiBody: Record<string, any> = {
      model: mapModel(anthropicBody.model),
      messages: openaiMessages,
      max_tokens: anthropicBody.max_tokens ?? 4096,
      temperature: anthropicBody.temperature ?? 0.7,
      stream: wantStream,
      chat_template_kwargs: { enable_thinking: process.env.ENABLE_THINKING === "1" },
    };
    if (process.env.REASONING_EFFORT) {
      openaiBody.reasoning_effort = process.env.REASONING_EFFORT;
    }

    const upstream = await fetch(upstreamBase + "/chat/completions", {
      method: "POST",
      headers: { "Content-Type": "application/json", Authorization: "Bearer fake" },
      body: JSON.stringify(openaiBody),
    });

    if (!upstream.ok) {
      const errText = await upstream.text();
      return new Response(errText, { status: upstream.status, headers: { "Content-Type": "application/json" } });
    }

    if (!wantStream) {
      const openaiResp = await upstream.json();
      const anthropicResp = {
        id: openaiResp.id,
        type: "message",
        role: "assistant",
        model: anthropicBody.model,
        content: [{
          type: "text",
          text: openaiResp.choices?.[0]?.message?.content || openaiResp.choices?.[0]?.message?.reasoning_content || "",
        }],
        usage: openaiResp.usage,
      };
      return Response.json(anthropicResp);
    }

    const encoder = new TextEncoder();
    const decoder = new TextDecoder();
    const id = "msg_" + Math.random().toString(36).slice(2);
    let outputTokens = 0;

    const stream = new ReadableStream({
      async start(controller) {
        controller.enqueue(
          encoder.encode(
            `event: message_start\ndata: ${JSON.stringify({
              type: "message_start",
              message: { id, type: "message", role: "assistant", model: anthropicBody.model, content: [], usage: { input_tokens: 0, output_tokens: 0 } },
            })}\n\n`
          )
        );
        controller.enqueue(
          encoder.encode(
            `event: content_block_start\ndata: ${JSON.stringify({
              type: "content_block_start",
              index: 0,
              content_block: { type: "text", text: "" },
            })}\n\n`
          )
        );

        const reader = upstream.body!.getReader();
        let buffer = "";

        try {
          while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            buffer += decoder.decode(value, { stream: true });

            const lines = buffer.split("\n");
            buffer = lines.pop() || "";

            for (const line of lines) {
              if (!line.startsWith("data: ")) continue;
              const data = line.slice(6);
              if (data === "[DONE]") continue;
              try {
                const chunk = JSON.parse(data);
                const delta = chunk.choices?.[0]?.delta;
                const text = delta?.content || delta?.reasoning_content || "";
                if (text) {
                  outputTokens++;
                  controller.enqueue(
                    encoder.encode(
                      `event: content_block_delta\ndata: ${JSON.stringify({
                        type: "content_block_delta",
                        index: 0,
                        delta: { type: "text_delta", text },
                      })}\n\n`
                    )
                  );
                }
              } catch {}
            }
          }

          controller.enqueue(
            encoder.encode(
              `event: content_block_stop\ndata: ${JSON.stringify({ type: "content_block_stop", index: 0 })}\n\n`
            )
          );
          controller.enqueue(
            encoder.encode(
              `event: message_delta\ndata: ${JSON.stringify({
                type: "message_delta",
                delta: { stop_reason: "end_turn" },
                usage: { output_tokens: outputTokens },
              })}\n\n`
            )
          );
          controller.enqueue(
            encoder.encode(
              `event: message_stop\ndata: ${JSON.stringify({ type: "message_stop" })}\n\n`
            )
          );
        } catch (e) {
          controller.error(e);
        } finally {
          controller.close();
        }
      },
    });

    return new Response(stream, {
      headers: {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
        Connection: "keep-alive",
      },
    });
  }

  return new Response("Not Found", { status: 404 });
}

if (import.meta.main) {
  const server = Bun.serve({
    port: PORT,
    fetch: (req) => handleRequest(req),
  });
  console.log(`Fast proxy running on http://localhost:${server.port}`);
}