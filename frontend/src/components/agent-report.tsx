"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import ReactMarkdown from "react-markdown";
import {
  ArrowLeft,
  BrainCircuit,
  CheckCircle2,
  KeyRound,
  LoaderCircle,
  Scale,
  ShieldCheck,
  Sparkles,
  Timer,
} from "lucide-react";
import { fetchRun, generateReport } from "@/lib/api";
import type { BenchmarkRun } from "@/lib/types";

export function AgentReport({ runId }: { runId: string }) {
  const [run, setRun] = useState<BenchmarkRun | null>(null);
  const [apiKey, setApiKey] = useState("");
  const [report, setReport] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    let cancelled = false;

    fetchRun(runId).then((nextRun) => {
      if (cancelled) return;
      setRun(nextRun);
      if (nextRun.agent_report) setReport(nextRun.agent_report);
    }).catch((reason: unknown) => {
      if (!cancelled) setError(reason instanceof Error ? reason.message : "Unable to load benchmark.");
    });

    return () => { cancelled = true; };
  }, [runId]);

  async function createReport() {
    if (!apiKey.trim()) return setError("Enter an Anthropic API key to generate the report.");
    try {
      setLoading(true);
      setError("");
      setReport(await generateReport(runId, apiKey));
      setApiKey("");
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Report generation failed.");
    } finally {
      setLoading(false);
    }
  }

  if (!run) return <main className="flex min-h-[60vh] items-center justify-center"><LoaderCircle className="animate-spin text-[var(--teal)]" /></main>;
  if (!run.result) return <main className="mx-auto max-w-4xl px-5 py-16"><h1 className="text-3xl font-bold">Benchmark results are not available.</h1><Link className="btn-secondary mt-5" href={`/dashboard/${runId}`}>Return to run</Link></main>;

  const entries = Object.entries(run.result.benchmark_matrix).sort((a, b) => b[1].f1_score - a[1].f1_score);
  const winner = entries[0];
  const runnerUp = entries[1];
  const fastest = entries.reduce((best, item) => item[1].embed_latency_ms < best[1].embed_latency_ms ? item : best);
  const qualityGap = runnerUp ? winner[1].f1_score - runnerUp[1].f1_score : 0;
  const confidence = qualityGap >= 0.08 ? "Strong separation" : qualityGap >= 0.03 ? "Moderate separation" : "Close result";

  return <main className="px-5 py-10 md:px-10"><div className="mx-auto max-w-7xl"><Link href={`/dashboard/${runId}`} className="inline-flex items-center gap-2 text-sm font-bold text-[var(--muted)] hover:text-[var(--ink)]"><ArrowLeft size={16} /> Back to dashboard</Link><div className="mt-7 grid gap-8 lg:grid-cols-[0.72fr_1.28fr]"><aside className="space-y-5 lg:sticky lg:top-24 lg:self-start"><div><p className="eyebrow text-[var(--coral)]">Decision intelligence</p><h1 className="mt-2 text-4xl font-extrabold">Agent analysis</h1><p className="mt-3 leading-7 text-[var(--muted)]">Measured evidence first, model interpretation second.</p></div><section className="panel overflow-hidden"><div className="bg-[var(--ink)] p-5 text-white"><p className="eyebrow text-white/55">Recommended configuration</p><h2 className="mt-3 text-xl font-bold leading-7">{winner[0]}</h2><div className="mt-5 flex gap-6"><span><strong className="metric-number block text-2xl">{winner[1].f1_score.toFixed(4)}</strong><small className="text-white/55">F1 score</small></span><span><strong className="metric-number block text-2xl">{winner[1].rougeL.toFixed(4)}</strong><small className="text-white/55">ROUGE-L</small></span></div></div><div className="divide-y divide-[var(--line)] p-5"><div className="flex gap-3 py-3"><ShieldCheck className="shrink-0 text-[var(--teal)]" size={19} /><div><strong className="text-sm">{confidence}</strong><p className="mt-1 text-xs text-[var(--muted)]">F1 lead of {qualityGap.toFixed(4)} over the runner-up.</p></div></div><div className="flex gap-3 py-3"><Timer className="shrink-0 text-[var(--amber)]" size={19} /><div><strong className="text-sm">Fastest alternative</strong><p className="mt-1 text-xs leading-5 text-[var(--muted)]">{fastest[0]} at {fastest[1].embed_latency_ms.toFixed(0)} ms.</p></div></div><div className="flex gap-3 py-3"><Scale className="shrink-0 text-[var(--coral)]" size={19} /><div><strong className="text-sm">Evidence scope</strong><p className="mt-1 text-xs text-[var(--muted)]">{run.result.qa_dataset.length} questions across {entries.length} configurations.</p></div></div></div></section>{!report && <section className="panel p-5"><div className="flex items-center gap-2"><KeyRound size={18} className="text-[var(--coral)]" /><h2 className="font-bold">Generate interpretation</h2></div><p className="mt-2 text-xs leading-5 text-[var(--muted)]">Claude will analyze measured rankings, score gaps, latency, cost, and evaluation limits. The key is used only for this request.</p><input className="control mt-4" type="password" placeholder="Anthropic API key" value={apiKey} onChange={(event) => setApiKey(event.target.value)} /><button type="button" onClick={createReport} disabled={loading} className="btn-primary mt-3 w-full disabled:opacity-65">{loading ? <><LoaderCircle className="animate-spin" size={16} /> Analyzing evidence</> : <><Sparkles size={16} /> Generate report</>}</button>{error && <p className="mt-3 text-sm font-semibold text-[var(--coral-dark)]">{error}</p>}</section>}</aside><section className="panel min-h-[640px] p-6 md:p-9">{report ? <div className="prose prose-neutral max-w-none [&_h2]:mt-9 [&_h2]:border-b [&_h2]:border-[var(--line)] [&_h2]:pb-3 [&_h2]:text-xl [&_li]:my-2 [&_p]:leading-7"><div className="mb-8 flex items-center gap-3 border-b border-[var(--line)] pb-5"><span className="flex size-10 items-center justify-center rounded-md bg-[var(--soft-teal)] text-[var(--teal)]"><BrainCircuit size={22} /></span><div><p className="eyebrow text-[var(--teal)]">Generated analysis</p><p className="text-sm text-[var(--muted)]">Grounded in this run&apos;s benchmark matrix</p></div></div><ReactMarkdown>{report}</ReactMarkdown></div> : <div className="flex min-h-[560px] flex-col items-center justify-center text-center"><BrainCircuit size={42} strokeWidth={1.4} className="text-[var(--teal)]" /><h2 className="mt-5 text-2xl font-bold">Evidence is ready for synthesis</h2><p className="mt-3 max-w-md leading-7 text-[var(--muted)]">Generate the report to receive a comparative explanation, production recommendation, confidence assessment, and the next validation step.</p><div className="mt-7 flex items-center gap-2 text-sm font-semibold text-[var(--teal)]"><CheckCircle2 size={17} /> Raw measurements remain visible alongside the analysis</div></div>}</section></div></div></main>;
}