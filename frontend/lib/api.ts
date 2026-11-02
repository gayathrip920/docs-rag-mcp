// Small helpers for talking to the Python API through the /api proxy.

export type Config = { ask_enabled: boolean; search_modes: string[] };

export type SearchResult = {
  page_path: string;
  heading: string;
  url: string;
  content: string;
  score: number;
};

export type Source = {
  n: number;
  page_path: string;
  heading: string;
  url: string;
  cited?: boolean;
};

export type DoneEvent = {
  type: "done";
  answer: string;
  found: boolean;
  citations_valid: boolean | null;
  sources: Source[];
  latency_ms: number;
  cost_usd: number | null;
  model: string;
};

export type StreamEvent =
  | { type: "sources"; sources: Source[] }
  | { type: "token"; text: string }
  | { type: "error"; message: string }
  | DoneEvent;

export type Metrics = {
  questions: number;
  found_rate: number | null;
  latency_ms_p50: number | null;
  latency_ms_p95: number | null;
  avg_cost_usd: number | null;
};

/** Only plain https links are shown as links. Anything else is ignored. */
export function safeHref(url: string): string | null {
  return url.startsWith("https://") ? url : null;
}

/** Makes a doc snippet easier to read: no link syntax, HTML tags or bold marks. Still plain text. */
export function cleanSnippet(text: string): string {
  return text
    .replace(/\[([^\]]+)\]\([^)]*\)/g, "$1")
    .replace(/<[^>]+>/g, "")
    .replace(/\*\*(.+?)\*\*/g, "$1")
    .replace(/\n{3,}/g, "\n\n")
    .trim();
}

async function errorMessage(res: Response): Promise<string> {
  try {
    const data = await res.json();
    if (typeof data.detail === "string") return data.detail;
    if (res.status === 422) return "That request was not valid. Check the text and try again.";
  } catch {
    // fall through to the generic message
  }
  return `Something went wrong (status ${res.status}).`;
}

async function getJson<T>(path: string): Promise<T> {
  const res = await fetch(`/api${path}`);
  if (!res.ok) throw new Error(await errorMessage(res));
  return res.json();
}

export const getConfig = () => getJson<Config>("/config");
export const getMetrics = () => getJson<Metrics>("/metrics");

export async function search(query: string, k = 5): Promise<SearchResult[]> {
  const res = await fetch("/api/search", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, k }),
  });
  if (!res.ok) throw new Error(await errorMessage(res));
  return (await res.json()).results;
}

/** Ask a question and receive the answer piece by piece. */
export async function* askStream(
  question: string,
  apiKey: string,
  model: string,
  signal?: AbortSignal,
): AsyncGenerator<StreamEvent> {
  const res = await fetch("/api/ask/stream", {
    method: "POST",
    headers: { "Content-Type": "application/json", "X-OpenRouter-Key": apiKey },
    body: JSON.stringify({ question, ...(model ? { model } : {}) }),
    signal,
  });
  if (!res.ok) throw new Error(await errorMessage(res));
  if (!res.body) throw new Error("The server sent no data.");

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const parts = buffer.split("\n\n");
    buffer = parts.pop() ?? "";
    for (const part of parts) {
      const line = part.split("\n").find((l) => l.startsWith("data: "));
      if (!line) continue;
      try {
        yield JSON.parse(line.slice(6)) as StreamEvent;
      } catch {
        // ignore a broken piece and keep reading
      }
    }
  }
}
