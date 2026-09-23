"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import {
  AlertCircle,
  ArrowRight,
  BarChart3,
  BrainCircuit,
  CheckCircle2,
  Download,
  FileSearch,
  Grid3X3,
  LoaderCircle,
  Sparkles,
} from "lucide-react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { embeddingExportUrl, exportUrl, fetchRun } from "@/lib/api";
import type { BenchmarkRun, Metrics, RawResult } from "@/lib/types";

type View = "overview" | "heatmap" | "qa" | "recommendations";

const metricLabels: Record<string, string> = {
  f1_score: "F1",
  rougeL: "ROUGE-L",
  bleu: "BLEU",
  context_precision: "Context precision",
  context_recall: "Context recall",
};

function shortConfig(config: string) {
  return config
    .replace("HuggingFace ", "HF ")
    .replace("OpenAI ", "")
    .replace(" | ", " / ");
}

function MetricCard({
  label,
  value,
  accent,
  detail,
}: {
  label: string;
  value: string;
  accent: string;
  detail: string;
}) {
  return (
    <div className="panel border-t-4 p-5" style={{ borderTopColor: accent }}>
      <p className="eyebrow text-[var(--muted)]">{label}</p>
      <p className="metric-number mt-2 text-3xl font-semibold">{value}</p>
      <p className="mt-2 truncate text-xs text-[var(--muted)]" title={detail}>
        {detail}
      </p>
    </div>
  );
}

function PendingRun({ run }: { run: BenchmarkRun }) {
  return (
    <main className="page-grid min-h-[calc(100vh-68px)] px-5 py-16 md:px-10">
      <div className="mx-auto max-w-3xl panel p-8 md:p-12">
        <div className="flex size-12 items-center justify-center rounded-md bg-[var(--soft-teal)] text-[var(--teal)]">
          <LoaderCircle className="animate-spin" />
        </div>
        <p className="eyebrow mt-8 text-[var(--teal)]">
          Run {run.id.slice(0, 8)}
        </p>
        <h1 className="mt-2 text-4xl font-extrabold">Benchmark in progress</h1>
        <p className="mt-4 text-lg text-[var(--muted)]">{run.stage}</p>
        <div className="mt-8 h-2 overflow-hidden rounded-full bg-[var(--line)]">
          <div className="h-full w-2/3 animate-pulse bg-[var(--coral)]" />
        </div>
        <div className="mt-8 grid gap-4 border-t border-[var(--line)] pt-6 sm:grid-cols-3">
          <span>
            <strong className="metric-number block text-xl">
              {run.configuration.chunk_strategies.length}
            </strong>
            <small className="text-[var(--muted)]">strategies</small>
          </span>
          <span>
            <strong className="metric-number block text-xl">
              {run.configuration.embedding_models.length}
            </strong>
            <small className="text-[var(--muted)]">models</small>
          </span>
          <span>
            <strong className="metric-number block text-xl">
              {run.configuration.question_count}
            </strong>
            <small className="text-[var(--muted)]">questions</small>
          </span>
        </div>
      </div>
    </main>
  );
}

function Overview({ matrix }: { matrix: Record<string, Metrics> }) {
  const data = Object.entries(matrix).map(([configuration, metrics]) => ({
    configuration: shortConfig(configuration),
    ...metrics,
  }));
  const hasRagas = data.some(
    (metrics) =>
      metrics.context_precision !== undefined ||
      metrics.context_recall !== undefined,
  );
  return (
    <div className="space-y-6">
      <section className="panel p-5 md:p-7">
        <div className="mb-6">
          <p className="eyebrow text-[var(--teal)]">Quality comparison</p>
          <h2 className="mt-1 text-xl font-bold">Core retrieval metrics</h2>
          <p className="mt-2 text-sm text-[var(--muted)]">
            {hasRagas
              ? "Ragas Context Precision and Context Recall are included."
              : "Ragas scores were not included in this run."}
          </p>
        </div>
        <div className="h-[390px]">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={data} margin={{ left: 8, right: 8, bottom: 70 }}>
              <CartesianGrid stroke="#e1e0d9" vertical={false} />
              <XAxis
                dataKey="configuration"
                angle={-32}
                textAnchor="end"
                interval={0}
                tick={{ fontSize: 10, fill: "#696d67" }}
              />
              <YAxis domain={[0, 1]} tick={{ fontSize: 11 }} />
              <Tooltip
                contentStyle={{ borderRadius: 6, borderColor: "#d8d8cf" }}
              />
              <Legend verticalAlign="top" height={34} />
              <Bar
                dataKey="f1_score"
                name="F1"
                fill="#087f72"
                radius={[3, 3, 0, 0]}
              />
              <Bar
                dataKey="rougeL"
                name="ROUGE-L"
                fill="#e65336"
                radius={[3, 3, 0, 0]}
              />
              <Bar
                dataKey="bleu"
                name="BLEU"
                fill="#d49735"
                radius={[3, 3, 0, 0]}
              />
              {hasRagas && (
                <Bar
                  dataKey="context_precision"
                  name="Ragas precision"
                  fill="#2864a6"
                  radius={[3, 3, 0, 0]}
                />
              )}
              {hasRagas && (
                <Bar
                  dataKey="context_recall"
                  name="Ragas recall"
                  fill="#7c5b9e"
                  radius={[3, 3, 0, 0]}
                />
              )}
            </BarChart>
          </ResponsiveContainer>
        </div>
      </section>
      <section className="panel overflow-x-auto">
        <table className="w-full min-w-[780px] text-left text-sm">
          <thead className="border-b border-[var(--line)] bg-[var(--paper)]">
            <tr>
              <th className="p-4">Configuration</th>
              {[
                "F1",
                "ROUGE-L",
                "BLEU",
                ...(hasRagas ? ["Ragas precision", "Ragas recall"] : []),
                "Latency",
                "Cost",
              ].map((item) => (
                <th key={item} className="p-4">
                  {item}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {Object.entries(matrix)
              .sort((a, b) => b[1].f1_score - a[1].f1_score)
              .map(([name, metrics], index) => (
                <tr
                  key={name}
                  className="border-b border-[var(--line)] last:border-0"
                >
                  <td className="p-4 font-semibold">
                    <span className="metric-number mr-3 text-xs text-[var(--muted)]">
                      {String(index + 1).padStart(2, "0")}
                    </span>
                    {name}
                  </td>
                  <td className="metric-number p-4 font-bold text-[var(--teal)]">
                    {metrics.f1_score.toFixed(4)}
                  </td>
                  <td className="metric-number p-4">
                    {metrics.rougeL.toFixed(4)}
                  </td>
                  <td className="metric-number p-4">
                    {metrics.bleu.toFixed(4)}
                  </td>
                  {hasRagas && (
                    <td className="metric-number p-4">
                      {metrics.context_precision?.toFixed(4) ?? "—"}
                    </td>
                  )}
                  {hasRagas && (
                    <td className="metric-number p-4">
                      {metrics.context_recall?.toFixed(4) ?? "—"}
                    </td>
                  )}
                  <td className="metric-number p-4">
                    {metrics.embed_latency_ms.toFixed(0)} ms
                  </td>
                  <td className="metric-number p-4">
                    ${metrics.embedding_cost_usd.toFixed(6)}
                  </td>
                </tr>
              ))}
          </tbody>
        </table>
      </section>
    </div>
  );
}

function EmbeddingDownloads({ run }: { run: BenchmarkRun }) {
  const summary = run.result?.embedding_summary;
  if (!summary) {
    return (
      <section className="panel mt-6 p-5 md:p-7">
        <p className="eyebrow text-[var(--muted)]">Embedding exports</p>
        <h2 className="mt-1 text-xl font-bold">No vectors retained</h2>
        <p className="mt-2 text-sm text-[var(--muted)]">
          Enable Keep embeddings for export before starting a run to download
          NPZ, FAISS, or JSONL files.
        </p>
      </section>
    );
  }

  return (
    <section className="panel mt-6 overflow-hidden">
      <div className="border-b border-[var(--line)] p-5 md:p-7">
        <p className="eyebrow text-[var(--teal)]">Embedding exports</p>
        <h2 className="mt-1 text-xl font-bold">Download retained vectors</h2>
        <p className="mt-2 text-sm text-[var(--muted)]">
          {summary.total_vectors} vectors retained for this backend session.
          ChromaDB storage is temporary; use these files for portable
          persistence.
        </p>
      </div>
      <div className="divide-y divide-[var(--line)]">
        {summary.configs.map((config) => (
          <div
            key={config.config}
            className="flex flex-col gap-4 p-5 md:flex-row md:items-center md:justify-between"
          >
            <div>
              <p className="text-sm font-bold">
                {config.model} / {config.strategy}
              </p>
              <p className="metric-number mt-1 text-xs text-[var(--muted)]">
                {config.num_chunks} chunks · {config.dimensions} dimensions ·{" "}
                {config.size_mb.toFixed(2)} MB
              </p>
            </div>
            <div className="flex flex-wrap gap-2">
              {(["npz", "faiss", "jsonl"] as const).map((format) => (
                <a
                  key={format}
                  className="btn-quiet"
                  href={embeddingExportUrl(run.id, config.config, format)}
                >
                  <Download size={14} /> {format.toUpperCase()}
                </a>
              ))}
            </div>
          </div>
        ))}
      </div>
    </section>
  );
}

function Heatmap({ matrix }: { matrix: Record<string, Metrics> }) {
  const [metric, setMetric] = useState("f1_score");
  const rows = Object.entries(matrix).map(([configuration, metrics]) => ({
    configuration,
    value: metrics[metric as keyof Metrics] as number,
  }));
  const max = Math.max(...rows.map((row) => row.value), 0.001);
  return (
    <section className="panel p-5 md:p-7">
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <p className="eyebrow text-[var(--teal)]">Metric intensity</p>
          <h2 className="mt-1 text-xl font-bold">Configuration heatmap</h2>
        </div>
        <select
          className="control max-w-52"
          value={metric}
          onChange={(event) => setMetric(event.target.value)}
        >
          {Object.entries(metricLabels)
            .filter(([key]) =>
              rows.some(({ configuration }) => key in matrix[configuration]),
            )
            .map(([key, label]) => (
              <option key={key} value={key}>
                {label}
              </option>
            ))}
        </select>
      </div>
      <div className="mt-8 grid gap-2 sm:grid-cols-2 lg:grid-cols-3">
        {rows
          .sort((a, b) => b.value - a.value)
          .map((row) => (
            <div
              key={row.configuration}
              className="min-h-28 rounded-md border border-[var(--line)] p-4"
              style={{
                backgroundColor: `color-mix(in srgb, var(--teal) ${Math.max(8, (row.value / max) * 88)}%, white)`,
              }}
            >
              <p className="text-xs font-bold leading-5">{row.configuration}</p>
              <p className="metric-number mt-4 text-2xl font-semibold">
                {row.value.toFixed(4)}
              </p>
            </div>
          ))}
      </div>
    </section>
  );
}

function QAInspector({ results }: { results: RawResult[] }) {
  const questions = Array.from(
    new Set(results.map((result) => result.question)),
  );
  const [question, setQuestion] = useState(questions[0] ?? "");
  const rows = results
    .filter((result) => result.question === question)
    .sort((a, b) => b.metrics.f1_score - a.metrics.f1_score);
  const selected = rows[0];
  return (
    <div className="space-y-6">
      <section className="panel p-5">
        <label className="eyebrow text-[var(--muted)]">
          Inspect question
          <select
            className="control mt-2 normal-case"
            value={question}
            onChange={(event) => setQuestion(event.target.value)}
          >
            {questions.map((item) => (
              <option key={item}>{item}</option>
            ))}
          </select>
        </label>
      </section>
      {selected && (
        <section className="grid gap-6 lg:grid-cols-2">
          <article className="panel p-5">
            <p className="eyebrow text-[var(--teal)]">Expected evidence</p>
            <p className="mt-4 whitespace-pre-wrap text-sm leading-7">
              {selected.expected_context}
            </p>
            <div className="mt-5 border-t border-[var(--line)] pt-4">
              <p className="eyebrow text-[var(--muted)]">Reference answer</p>
              <p className="mt-2 text-sm leading-6">{selected.ground_truth}</p>
            </div>
          </article>
          <article className="panel p-5">
            <p className="eyebrow text-[var(--coral)]">Top retrieved context</p>
            <p className="mt-4 max-h-80 overflow-y-auto whitespace-pre-wrap text-sm leading-7">
              {selected.retrieved_context}
            </p>
          </article>
        </section>
      )}
      <section className="panel overflow-x-auto">
        <table className="w-full min-w-[680px] text-left text-sm">
          <thead className="border-b border-[var(--line)] bg-[var(--paper)]">
            <tr>
              <th className="p-4">Configuration</th>
              <th className="p-4">F1</th>
              <th className="p-4">ROUGE-L</th>
              <th className="p-4">BLEU</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => (
              <tr
                key={`${row.embedding_model}-${row.chunk_strategy}`}
                className="border-b border-[var(--line)]"
              >
                <td className="p-4 font-semibold">
                  {row.embedding_model} / {row.chunk_strategy}
                </td>
                <td className="metric-number p-4">
                  {row.metrics.f1_score.toFixed(4)}
                </td>
                <td className="metric-number p-4">
                  {row.metrics.rougeL.toFixed(4)}
                </td>
                <td className="metric-number p-4">
                  {row.metrics.bleu.toFixed(4)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>
    </div>
  );
}

function Recommendations({ run }: { run: BenchmarkRun }) {
  const result = run.result!;
  return (
    <div className="grid gap-6 lg:grid-cols-[1.2fr_0.8fr]">
      <section className="panel p-5 md:p-7">
        <p className="eyebrow text-[var(--coral)]">Evidence-led findings</p>
        <h2 className="mt-1 text-2xl font-bold">
          What the benchmark indicates
        </h2>
        <div className="mt-6 space-y-3">
          {result.recommendations.insights.map((insight) => (
            <div
              key={insight}
              className="flex gap-3 rounded-md border border-[var(--line)] bg-[var(--paper)] p-4"
            >
              <CheckCircle2
                className="mt-0.5 shrink-0 text-[var(--teal)]"
                size={18}
              />
              <p
                className="text-sm leading-6"
                dangerouslySetInnerHTML={{ __html: insight }}
              />
            </div>
          ))}
        </div>
        <Link href={`/reports/${run.id}`} className="btn-primary mt-6">
          Open agent analysis <ArrowRight size={16} />
        </Link>
      </section>
      <div className="space-y-6">
        <section className="panel p-5">
          <p className="eyebrow text-[var(--muted)]">Model ranking</p>
          {result.recommendations.model_ranking.map((item, index) => (
            <div
              key={item.Model}
              className="mt-4 flex items-center justify-between border-b border-[var(--line)] pb-3 text-sm last:border-0"
            >
              <span>
                <strong className="metric-number mr-2 text-[var(--muted)]">
                  {index + 1}
                </strong>
                {item.Model}
              </span>
              <strong className="metric-number">
                {item["Avg Score"].toFixed(4)}
              </strong>
            </div>
          ))}
        </section>
        <section className="panel p-5">
          <p className="eyebrow text-[var(--muted)]">Chunking ranking</p>
          {result.recommendations.chunking_ranking.map((item, index) => (
            <div
              key={item.Strategy}
              className="mt-4 flex items-center justify-between border-b border-[var(--line)] pb-3 text-sm last:border-0"
            >
              <span>
                <strong className="metric-number mr-2 text-[var(--muted)]">
                  {index + 1}
                </strong>
                {item.Strategy}
              </span>
              <strong className="metric-number">
                {item["Avg Score"].toFixed(4)}
              </strong>
            </div>
          ))}
        </section>
      </div>
    </div>
  );
}

export function RunDashboard({ runId }: { runId: string }) {
  const [run, setRun] = useState<BenchmarkRun | null>(null);
  const [view, setView] = useState<View>("overview");
  const [loadError, setLoadError] = useState("");

  useEffect(() => {
    let cancelled = false;
    let timer: number | undefined;

    function loadRun() {
      fetchRun(runId)
        .then((nextRun) => {
          if (cancelled) return;
          setRun(nextRun);
          if (["queued", "running"].includes(nextRun.status)) {
            timer = window.setTimeout(loadRun, 1800);
          }
        })
        .catch((reason: unknown) => {
          if (!cancelled)
            setLoadError(
              reason instanceof Error ? reason.message : "Unable to load run.",
            );
        });
    }

    loadRun();
    return () => {
      cancelled = true;
      if (timer) window.clearTimeout(timer);
    };
  }, [runId]);

  if (loadError)
    return (
      <main className="mx-auto max-w-4xl px-5 py-16">
        <div className="panel border-[var(--coral)] p-6">
          <AlertCircle className="text-[var(--coral)]" />
          <h1 className="mt-4 text-2xl font-bold">Unable to load benchmark</h1>
          <p className="mt-2 text-[var(--muted)]">{loadError}</p>
        </div>
      </main>
    );
  if (!run)
    return (
      <main className="flex min-h-[60vh] items-center justify-center">
        <LoaderCircle className="animate-spin text-[var(--teal)]" />
      </main>
    );
  if (run.status === "queued" || run.status === "running")
    return <PendingRun run={run} />;
  if (run.status === "failed")
    return (
      <main className="mx-auto max-w-4xl px-5 py-16">
        <div className="panel border-[var(--coral)] p-7">
          <AlertCircle className="text-[var(--coral)]" />
          <p className="eyebrow mt-5 text-[var(--coral)]">Benchmark failed</p>
          <h1 className="mt-2 text-3xl font-bold">
            The run could not complete
          </h1>
          <p className="mt-4 text-[var(--muted)]">{run.error}</p>
          <Link href="/workspace" className="btn-primary mt-6">
            Return to workspace
          </Link>
        </div>
      </main>
    );

  const matrix = run.result!.benchmark_matrix;
  const entries = Object.entries(matrix);
  const bestF1 = entries.reduce((best, item) =>
    item[1].f1_score > best[1].f1_score ? item : best,
  );
  const bestRouge = entries.reduce((best, item) =>
    item[1].rougeL > best[1].rougeL ? item : best,
  );
  const fastest = entries.reduce((best, item) =>
    item[1].embed_latency_ms < best[1].embed_latency_ms ? item : best,
  );
  const navItems: Array<[View, string, typeof BarChart3]> = [
    ["overview", "Overview", BarChart3],
    ["heatmap", "Heatmap", Grid3X3],
    ["qa", "QA inspector", FileSearch],
    ["recommendations", "Recommendations", BrainCircuit],
  ];

  return (
    <main className="px-5 py-9 md:px-10">
      <div className="mx-auto max-w-7xl">
        <div className="flex flex-wrap items-end justify-between gap-5 border-b border-[var(--line)] pb-7">
          <div>
            <div className="flex items-center gap-2">
              <span className="rounded-full bg-[var(--soft-teal)] px-3 py-1 text-xs font-bold text-[var(--teal)]">
                Complete
              </span>
              <span className="eyebrow text-[var(--muted)]">
                Run {run.id.slice(0, 8)}
              </span>
            </div>
            <h1 className="mt-3 text-4xl font-extrabold">Benchmark results</h1>
            <p className="mt-2 text-sm text-[var(--muted)]">
              {run.file_name} · {run.result!.doc_type} document ·{" "}
              {entries.length} configurations
            </p>
          </div>
          <div className="flex flex-wrap gap-2">
            <a className="btn-quiet" href={exportUrl(run.id, "csv")}>
              <Download size={15} /> CSV
            </a>
            <a className="btn-quiet" href={exportUrl(run.id, "json")}>
              <Download size={15} /> JSON
            </a>
            <Link className="btn-primary" href={`/reports/${run.id}`}>
              <Sparkles size={15} /> Agent report
            </Link>
          </div>
        </div>
        <div className="mt-6 grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
          <MetricCard
            label="Configurations"
            value={String(entries.length)}
            accent="var(--ink)"
            detail={`${run.configuration.embedding_models.length} models × ${run.configuration.chunk_strategies.length} strategies`}
          />
          <MetricCard
            label="Best F1"
            value={bestF1[1].f1_score.toFixed(4)}
            accent="var(--teal)"
            detail={bestF1[0]}
          />
          <MetricCard
            label="Best ROUGE-L"
            value={bestRouge[1].rougeL.toFixed(4)}
            accent="var(--coral)"
            detail={bestRouge[0]}
          />
          <MetricCard
            label="Fastest embedding"
            value={`${fastest[1].embed_latency_ms.toFixed(0)} ms`}
            accent="var(--amber)"
            detail={fastest[0]}
          />
        </div>
        <div className="my-6 flex gap-1 overflow-x-auto border-b border-[var(--line)]">
          {navItems.map(([key, label, Icon]) => (
            <button
              key={key}
              onClick={() => setView(key)}
              className={`flex min-h-11 shrink-0 items-center gap-2 border-b-2 px-4 text-sm font-bold ${view === key ? "border-[var(--coral)] text-[var(--ink)]" : "border-transparent text-[var(--muted)]"}`}
            >
              <Icon size={16} />
              {label}
            </button>
          ))}
        </div>
        {view === "overview" && (
          <>
            <Overview matrix={matrix} />
            <EmbeddingDownloads run={run} />
          </>
        )}
        {view === "heatmap" && <Heatmap matrix={matrix} />}
        {view === "qa" && <QAInspector results={run.result!.raw_results} />}
        {view === "recommendations" && <Recommendations run={run} />}
      </div>
    </main>
  );
}
