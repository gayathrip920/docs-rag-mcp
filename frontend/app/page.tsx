"use client";

import { useEffect, useRef, useState } from "react";
import AnswerText from "@/components/AnswerText";
import SourceLink from "@/components/SourceLink";
import {
  askStream,
  cleanSnippet,
  getConfig,
  search,
  type Config,
  type DoneEvent,
  type SearchResult,
  type Source,
} from "@/lib/api";

const KEY_STORAGE = "openrouter-key"; // sessionStorage: cleared when the tab closes

export default function Home() {
  const [config, setConfig] = useState<Config | null>(null);
  const [question, setQuestion] = useState("");
  const [apiKey, setApiKey] = useState("");
  const [model, setModel] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [results, setResults] = useState<SearchResult[] | null>(null);
  const [streamed, setStreamed] = useState("");
  const [sources, setSources] = useState<Source[]>([]);
  const [done, setDone] = useState<DoneEvent | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  useEffect(() => {
    getConfig()
      .then(setConfig)
      .catch(() => setConfig({ ask_enabled: false, search_modes: [] }));
    try {
      setApiKey(sessionStorage.getItem(KEY_STORAGE) ?? "");
    } catch {
      // storage can be blocked, the key box still works
    }
  }, []);

  function changeKey(value: string) {
    setApiKey(value);
    try {
      sessionStorage.setItem(KEY_STORAGE, value);
    } catch {
      // ignore
    }
  }

  function reset() {
    setError("");
    setResults(null);
    setStreamed("");
    setSources([]);
    setDone(null);
  }

  async function runSearch() {
    reset();
    setBusy(true);
    try {
      setResults(await search(question.trim()));
    } catch (e) {
      setError(e instanceof Error ? e.message : "Something went wrong.");
    } finally {
      setBusy(false);
    }
  }

  async function runAsk() {
    reset();
    setBusy(true);
    abortRef.current = new AbortController();
    try {
      for await (const event of askStream(
        question.trim(),
        apiKey.trim(),
        model.trim(),
        abortRef.current.signal,
      )) {
        if (event.type === "sources") setSources(event.sources);
        else if (event.type === "token") setStreamed((s) => s + event.text);
        else if (event.type === "error") setError(event.message);
        else if (event.type === "done") {
          setDone(event);
          setSources(event.sources);
        }
      }
    } catch (e) {
      if (!(e instanceof DOMException && e.name === "AbortError")) {
        setError(e instanceof Error ? e.message : "Something went wrong.");
      }
    } finally {
      setBusy(false);
    }
  }

  const askEnabled = config?.ask_enabled ?? false;
  const canSubmit = question.trim().length > 0 && !busy;
  const shownAnswer = done ? done.answer : streamed;

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-semibold">Search the FastAPI docs</h1>
        <p className="mt-1 text-slate-600 dark:text-slate-400">
          Ask in your own words. You get the best matching parts of the docs, with links.
        </p>
      </div>

      <form
        className="space-y-3"
        onSubmit={(e) => {
          e.preventDefault();
          if (canSubmit) runSearch();
        }}
      >
        <label htmlFor="question" className="block text-sm font-medium">
          Your question
        </label>
        <input
          id="question"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          maxLength={500}
          placeholder="How do I let users upload a file?"
          className="w-full rounded-md border border-slate-300 bg-white px-3 py-2 dark:border-slate-700 dark:bg-slate-900"
        />

        {askEnabled && (
          <div className="grid gap-3 rounded-md border border-slate-200 p-3 sm:grid-cols-2 dark:border-slate-800">
            <div className="sm:col-span-2">
              <label htmlFor="key" className="block text-sm font-medium">
                Your OpenRouter API key (needed for written answers)
              </label>
              <input
                id="key"
                type="password"
                autoComplete="off"
                spellCheck={false}
                value={apiKey}
                onChange={(e) => changeKey(e.target.value)}
                placeholder="sk-or-..."
                className="mt-1 w-full rounded-md border border-slate-300 bg-white px-3 py-2 font-mono text-sm dark:border-slate-700 dark:bg-slate-900"
              />
              <p className="mt-1 text-xs text-slate-500">
                Used only for your question. The server never saves it. It stays in this browser tab
                and is gone when you close it.
              </p>
            </div>
            <div className="sm:col-span-2">
              <label htmlFor="model" className="block text-sm font-medium">
                Model (optional)
              </label>
              <input
                id="model"
                value={model}
                onChange={(e) => setModel(e.target.value)}
                placeholder="openai/gpt-4o-mini"
                className="mt-1 w-full rounded-md border border-slate-300 bg-white px-3 py-2 text-sm dark:border-slate-700 dark:bg-slate-900"
              />
            </div>
          </div>
        )}

        <div className="flex flex-wrap gap-2">
          <button
            type="submit"
            disabled={!canSubmit}
            className="rounded-md bg-slate-900 px-4 py-2 text-sm font-medium text-white disabled:opacity-50 dark:bg-slate-100 dark:text-slate-900"
          >
            {busy ? "Working..." : "Search docs"}
          </button>
          {askEnabled && (
            <button
              type="button"
              disabled={!canSubmit || apiKey.trim().length === 0}
              onClick={runAsk}
              className="rounded-md border border-slate-300 px-4 py-2 text-sm font-medium disabled:opacity-50 dark:border-slate-700"
            >
              Ask for an answer
            </button>
          )}
          {busy && abortRef.current && (
            <button
              type="button"
              onClick={() => abortRef.current?.abort()}
              className="rounded-md px-4 py-2 text-sm text-slate-600 underline"
            >
              Stop
            </button>
          )}
        </div>
      </form>

      <div aria-live="polite" className="space-y-4">
        {error && (
          <p role="alert" className="rounded-md bg-red-50 p-3 text-sm text-red-800 dark:bg-red-950 dark:text-red-200">
            {error}
          </p>
        )}

        {results && results.length === 0 && <p>No matches found.</p>}
        {results && results.length > 0 && (
          <ul className="space-y-4">
            {results.map((r, i) => (
              <li key={`${r.url}-${i}`} className="rounded-md border border-slate-200 p-4 dark:border-slate-800">
                <h2 className="font-medium">
                  <SourceLink url={r.url}>{r.heading}</SourceLink>
                </h2>
                <p className="mt-2 whitespace-pre-wrap text-sm text-slate-700 dark:text-slate-300">
                  {cleanSnippet(r.content.split("\n\n").slice(1).join("\n\n")).slice(0, 500)}
                  {r.content.length > 500 ? "..." : ""}
                </p>
              </li>
            ))}
          </ul>
        )}

        {(shownAnswer || done) && (
          <section className="space-y-3 rounded-md border border-slate-200 p-4 dark:border-slate-800">
            <h2 className="font-medium">Answer</h2>
            <AnswerText text={shownAnswer} sourceCount={sources.length} />
            {done && done.citations_valid === false && (
              <p className="text-sm text-amber-700 dark:text-amber-300">
                Warning: this answer did not cite its sources properly. Check the links below.
              </p>
            )}
            {done && (
              <p className="text-xs text-slate-500">
                {done.model} · {(done.latency_ms / 1000).toFixed(1)}s
                {done.cost_usd != null && ` · $${done.cost_usd.toFixed(5)}`}
              </p>
            )}
          </section>
        )}

        {sources.length > 0 && (
          <section>
            <h2 className="mb-2 text-sm font-medium">Sources</h2>
            <ol className="space-y-1 text-sm">
              {sources.map((s) => (
                <li key={s.n} id={`source-${s.n}`} className="scroll-mt-4">
                  [{s.n}] <SourceLink url={s.url}>{s.heading}</SourceLink>
                  {s.cited && <span className="ml-2 text-xs text-slate-500">cited</span>}
                </li>
              ))}
            </ol>
          </section>
        )}
      </div>
    </div>
  );
}
