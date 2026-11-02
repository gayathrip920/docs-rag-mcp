"use client";

import { useEffect, useState } from "react";
import { getMetrics, type Metrics } from "@/lib/api";

function Card({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-md border border-slate-200 p-4 dark:border-slate-800">
      <dt className="text-sm text-slate-500">{label}</dt>
      <dd className="mt-1 text-2xl font-semibold">{value}</dd>
    </div>
  );
}

const ms = (v: number | null) => (v == null ? "no data" : `${Math.round(v)} ms`);

export default function MetricsPage() {
  const [data, setData] = useState<Metrics | null>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    getMetrics()
      .then(setData)
      .catch((e) => setError(e instanceof Error ? e.message : "Could not load metrics."));
  }, []);

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-semibold">Metrics</h1>
      <p className="text-slate-600 dark:text-slate-400">
        Numbers from questions that asked for a written answer. Searches alone are not counted.
      </p>
      {error && <p role="alert" className="text-red-700 dark:text-red-300">{error}</p>}
      {!data && !error && <p>Loading...</p>}
      {data && (
        <dl className="grid gap-4 sm:grid-cols-2">
          <Card label="Questions answered" value={String(data.questions)} />
          <Card
            label="Answer found in docs"
            value={data.found_rate == null ? "no data" : `${Math.round(data.found_rate * 100)}%`}
          />
          <Card label="Median time (p50)" value={ms(data.latency_ms_p50)} />
          <Card label="Slowest 5% (p95)" value={ms(data.latency_ms_p95)} />
          <Card
            label="Average cost per question"
            value={data.avg_cost_usd == null ? "no data" : `$${data.avg_cost_usd.toFixed(5)}`}
          />
        </dl>
      )}
    </div>
  );
}
