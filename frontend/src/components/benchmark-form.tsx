"use client";

import { FormEvent, useState } from "react";
import { useRouter } from "next/navigation";
import {
  ArrowRight,
  Check,
  ChevronDown,
  FileText,
  KeyRound,
  Layers3,
  LoaderCircle,
  Plus,
  UploadCloud,
  X,
} from "lucide-react";
import { createRun } from "@/lib/api";
import type { AppConfig } from "@/lib/types";

type Props = { config: AppConfig };

const providerLabels = {
  huggingface: "Local / HuggingFace",
  openai: "OpenAI",
  cohere: "Cohere",
};

type Provider = keyof typeof providerLabels;

const customModelPrefixes: Record<Provider, string> = {
  huggingface: "HuggingFace Custom:",
  openai: "OpenAI Custom:",
  cohere: "Cohere Custom:",
};

function modelProvider(model: string): Provider {
  if (model.startsWith("OpenAI")) return "openai";
  if (model.startsWith("Cohere")) return "cohere";
  return "huggingface";
}

function Toggle({ checked, onChange, label, description }: { checked: boolean; onChange: (value: boolean) => void; label: string; description: string }) {
  return (
    <label className="flex cursor-pointer items-start gap-3 py-2">
      <button type="button" role="switch" aria-checked={checked} onClick={() => onChange(!checked)} className={`mt-0.5 flex h-6 w-11 shrink-0 items-center rounded-full p-0.5 transition ${checked ? "bg-[var(--teal)]" : "bg-[#c9cbc4]"}`}>
        <span className={`size-5 rounded-full bg-white transition-transform ${checked ? "translate-x-5" : "translate-x-0"}`} />
      </button>
      <span><strong className="block text-sm">{label}</strong><span className="mt-0.5 block text-xs leading-5 text-[var(--muted)]">{description}</span></span>
    </label>
  );
}

function Choice({ active, label, onClick }: { active: boolean; label: string; onClick: () => void }) {
  return (
    <button type="button" onClick={onClick} className={`flex min-h-11 items-center justify-between rounded-md border px-3 py-2 text-left text-sm font-semibold ${active ? "border-[var(--teal)] bg-[var(--soft-teal)] text-[var(--teal)]" : "border-[var(--line)] bg-white hover:border-[var(--ink)]"}`}>
      <span>{label}</span>{active && <Check size={15} />}
    </button>
  );
}

export function BenchmarkForm({ config }: Props) {
  const router = useRouter();
  const [document, setDocument] = useState<File | null>(null);
  const [groundTruth, setGroundTruth] = useState<File | null>(null);
  const [useSynthetic, setUseSynthetic] = useState(true);
  const [questionCount, setQuestionCount] = useState(15);
  const [topK, setTopK] = useState(config.default_top_k);
  const [chunkers, setChunkers] = useState<string[]>(["Recursive Character", "Sentence Window"]);
  const [models, setModels] = useState<string[]>([config.models.huggingface[0]]);
  const [customProvider, setCustomProvider] = useState<Provider>("huggingface");
  const [customModelId, setCustomModelId] = useState("");
  const [chunkParams, setChunkParams] = useState<Record<string, Record<string, number>>>({});
  const [useRagas, setUseRagas] = useState(false);
  const [persistEmbeddings, setPersistEmbeddings] = useState(false);
  const [useVectorDb, setUseVectorDb] = useState(false);
  const [keys, setKeys] = useState({ anthropic: "", openai: "", cohere: "", hf: "" });
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState("");

  const toggle = (value: string, values: string[], setter: (next: string[]) => void) => {
    setter(values.includes(value) ? values.filter((item) => item !== value) : [...values, value]);
  };

  const updateParam = (strategy: string, key: string, value: number) => {
    setChunkParams((current) => ({
      ...current,
      [strategy]: { ...current[strategy], [key]: value },
    }));
  };

  const addCustomModel = () => {
    const modelId = customModelId.trim();
    if (!modelId) {
      setError("Enter a model ID before adding a custom model.");
      return;
    }
    const model = `${customModelPrefixes[customProvider]} ${modelId}`;
    if (!models.includes(model)) setModels((current) => [...current, model]);
    setCustomModelId("");
    setError("");
  };

  const selectedProviders = new Set(models.map(modelProvider));
  const needsAnthropicKey = useSynthetic || useRagas;
  const customModels = models.filter((model) => model.includes(" Custom:"));

  async function submit(event: FormEvent) {
    event.preventDefault();
    setError("");
    if (!document) return setError("Upload a source document to continue.");
    if (!chunkers.length || !models.length) return setError("Select at least one chunking strategy and embedding model.");
    if (!useSynthetic && !groundTruth) return setError("Upload a ground-truth JSON or CSV dataset.");
    if (needsAnthropicKey && !keys.anthropic.trim()) return setError("An Anthropic API key is required for synthetic QA or Ragas evaluation.");
    if (selectedProviders.has("openai") && !keys.openai.trim()) return setError("An OpenAI API key is required for the selected OpenAI models.");
    if (selectedProviders.has("cohere") && !keys.cohere.trim()) return setError("A Cohere API key is required for the selected Cohere models.");

    const formData = new FormData();
    formData.append("document", document);
    if (groundTruth) formData.append("ground_truth", groundTruth);
    formData.append("use_synthetic", String(useSynthetic));
    formData.append("question_count", String(questionCount));
    formData.append("top_k", String(topK));
    formData.append("chunk_strategies", JSON.stringify(chunkers));
    formData.append("embedding_models", JSON.stringify(models));
    formData.append("chunk_params", JSON.stringify(chunkParams));
    formData.append("use_ragas", String(useRagas));
    formData.append("persist_embeddings", String(persistEmbeddings));
    formData.append("use_vector_db", String(useVectorDb));
    formData.append("anthropic_key", keys.anthropic);
    formData.append("openai_key", keys.openai);
    formData.append("cohere_key", keys.cohere);
    formData.append("hf_token", keys.hf);

    try {
      setSubmitting(true);
      const run = await createRun(formData);
      router.push(`/dashboard/${run.id}`);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Unable to create benchmark run.");
      setSubmitting(false);
    }
  }

  return (
    <main className="px-5 py-10 md:px-10">
      <form onSubmit={submit} className="mx-auto max-w-7xl">
        <div className="mb-10 grid gap-5 border-b border-[var(--line)] pb-8 md:grid-cols-[1fr_auto] md:items-end">
          <div><p className="eyebrow text-[var(--coral)]">New evaluation run</p><h1 className="mt-2 text-4xl font-extrabold md:text-5xl">Benchmark workspace</h1><p className="mt-3 max-w-2xl text-[var(--muted)]">Define the evidence set, retrieval architecture, and evaluation depth. Your credentials are sent only to the local Python service for this run.</p></div>
          <div className="flex gap-6 text-sm"><span><strong className="metric-number block text-2xl">{chunkers.length * models.length}</strong>configurations</span><span><strong className="metric-number block text-2xl">{questionCount}</strong>queries</span></div>
        </div>

        <div className="grid gap-6 lg:grid-cols-[1fr_350px]">
          <div className="space-y-6">
            <section className="panel p-5 md:p-7">
              <div className="mb-6 flex items-start gap-3"><FileText className="mt-0.5 text-[var(--coral)]" /><div><h2 className="text-xl font-bold">Source and evidence</h2><p className="mt-1 text-sm text-[var(--muted)]">Upload the content to evaluate and choose how ground truth is prepared.</p></div></div>
              <label className="flex min-h-36 cursor-pointer flex-col items-center justify-center rounded-md border border-dashed border-[var(--line)] bg-[var(--paper)] p-5 text-center hover:border-[var(--teal)]">
                <UploadCloud className="mb-3 text-[var(--teal)]" />
                <strong className="text-sm">{document?.name ?? "Choose a PDF, text, Markdown, Excel, or CSV file"}</strong>
                <span className="mt-1 text-xs text-[var(--muted)]">The file remains local to the benchmark service.</span>
                <input className="sr-only" type="file" accept=".pdf,.txt,.md,.xlsx,.csv" onChange={(event) => setDocument(event.target.files?.[0] ?? null)} />
              </label>
              <div className="mt-5 grid gap-4 md:grid-cols-2">
                <Choice active={useSynthetic} label="Generate synthetic QA" onClick={() => setUseSynthetic(true)} />
                <Choice active={!useSynthetic} label="Use existing QA dataset" onClick={() => setUseSynthetic(false)} />
              </div>
              {useSynthetic ? (
                <div className="mt-5"><label className="text-sm font-bold">Question count <span className="metric-number ml-2 text-[var(--teal)]">{questionCount}</span></label><input className="mt-3 w-full accent-[var(--teal)]" type="range" min="1" max={config.max_synthetic_questions} value={questionCount} onChange={(event) => setQuestionCount(Number(event.target.value))} /></div>
              ) : (
                <label className="mt-5 block text-sm font-bold">Ground-truth dataset<input className="control mt-2" type="file" accept=".json,.csv" onChange={(event) => setGroundTruth(event.target.files?.[0] ?? null)} /></label>
              )}
            </section>

            <section className="panel p-5 md:p-7">
              <div className="mb-6 flex items-start gap-3"><Layers3 className="mt-0.5 text-[var(--coral)]" /><div><h2 className="text-xl font-bold">Retrieval architecture</h2><p className="mt-1 text-sm text-[var(--muted)]">Select multiple options to create a complete cross-product benchmark.</p></div></div>
              <h3 className="eyebrow mb-3 text-[var(--muted)]">Chunking strategies</h3>
              <div className="grid gap-2 sm:grid-cols-2 xl:grid-cols-3">{config.chunking_strategies.map((strategy) => <Choice key={strategy} active={chunkers.includes(strategy)} label={strategy} onClick={() => toggle(strategy, chunkers, setChunkers)} />)}</div>
              <div className="mt-5 space-y-2">{chunkers.filter((strategy) => config.chunk_strategy_params[strategy]).map((strategy) => (
                <details key={strategy} className="rounded-md border border-[var(--line)] bg-[var(--paper)] px-4 py-3">
                  <summary className="flex cursor-pointer list-none items-center justify-between text-sm font-bold">Tune {strategy}<ChevronDown size={16} /></summary>
                  <div className="mt-4 grid gap-4 sm:grid-cols-2">{Object.entries(config.chunk_strategy_params[strategy]).map(([key, definition]) => <label key={key} className="text-xs font-bold text-[var(--muted)]">{definition.label}<input className="control mt-1" type="number" min={definition.min} max={definition.max} step={definition.step} value={chunkParams[strategy]?.[key] ?? definition.default} onChange={(event) => updateParam(strategy, key, Number(event.target.value))} /></label>)}</div>
                </details>
              ))}</div>

              <h3 className="eyebrow mb-3 mt-8 text-[var(--muted)]">Embedding models</h3>
              {Object.entries(config.models).map(([provider, providerModels]) => <div key={provider} className="mb-5"><p className="mb-2 text-sm font-bold">{providerLabels[provider as keyof typeof providerLabels]}</p><div className="grid gap-2 sm:grid-cols-2 xl:grid-cols-3">{providerModels.map((model) => <Choice key={model} active={models.includes(model)} label={model} onClick={() => toggle(model, models, setModels)} />)}</div></div>)}
              <div className="mt-7 border-t border-[var(--line)] pt-6">
                <h3 className="text-sm font-bold">Add another embedding model</h3>
                <p className="mt-1 text-xs leading-5 text-[var(--muted)]">Use a Hugging Face model ID or an embedding model ID enabled for your OpenAI or Cohere account.</p>
                <div className="mt-3 grid gap-2 sm:grid-cols-[160px_1fr_auto]">
                  <select className="control" value={customProvider} onChange={(event) => setCustomProvider(event.target.value as Provider)} aria-label="Custom model provider">
                    {Object.entries(providerLabels).map(([provider, label]) => <option key={provider} value={provider}>{label}</option>)}
                  </select>
                  <input className="control" value={customModelId} onChange={(event) => setCustomModelId(event.target.value)} placeholder={customProvider === "huggingface" ? "org/model-name" : "model-id"} aria-label="Custom model ID" />
                  <button className="btn-secondary" type="button" onClick={addCustomModel}><Plus size={16} /> Add model</button>
                </div>
                {customModels.length > 0 && <div className="mt-3 flex flex-wrap gap-2">{customModels.map((model) => <span key={model} className="inline-flex min-h-9 items-center gap-2 rounded-md border border-[var(--teal)] bg-[var(--soft-teal)] px-3 text-xs font-bold text-[var(--teal)]">{model}<button type="button" onClick={() => setModels((current) => current.filter((item) => item !== model))} aria-label={`Remove ${model}`} title={`Remove ${model}`}><X size={14} /></button></span>)}</div>}
              </div>
            </section>
          </div>

          <aside className="space-y-6 lg:sticky lg:top-24 lg:self-start">
            <section className="panel p-5">
              <div className="mb-5 flex items-center gap-2"><KeyRound size={19} className="text-[var(--coral)]" /><h2 className="font-bold">Provider credentials</h2></div>
              <div className="space-y-4">
                {needsAnthropicKey && <label className="block text-xs font-bold text-[var(--muted)]">Anthropic API key <span className="text-[var(--coral)]">Required</span><input className="control mt-1" type="password" autoComplete="off" value={keys.anthropic} onChange={(event) => setKeys((current) => ({ ...current, anthropic: event.target.value }))} /></label>}
                {selectedProviders.has("openai") && <label className="block text-xs font-bold text-[var(--muted)]">OpenAI API key <span className="text-[var(--coral)]">Required</span><input className="control mt-1" type="password" autoComplete="off" value={keys.openai} onChange={(event) => setKeys((current) => ({ ...current, openai: event.target.value }))} /></label>}
                {selectedProviders.has("cohere") && <label className="block text-xs font-bold text-[var(--muted)]">Cohere API key <span className="text-[var(--coral)]">Required</span><input className="control mt-1" type="password" autoComplete="off" value={keys.cohere} onChange={(event) => setKeys((current) => ({ ...current, cohere: event.target.value }))} /></label>}
                {selectedProviders.has("huggingface") && <label className="block text-xs font-bold text-[var(--muted)]">Hugging Face token <span className="font-normal">Optional for public models</span><input className="control mt-1" type="password" autoComplete="off" value={keys.hf} onChange={(event) => setKeys((current) => ({ ...current, hf: event.target.value }))} /></label>}
                {!needsAnthropicKey && selectedProviders.size === 0 && <p className="text-sm text-[var(--muted)]">No provider credentials are needed.</p>}
              </div>
            </section>
            <section className="panel p-5">
              <h2 className="mb-4 font-bold">Evaluation depth</h2>
              <label className="block text-xs font-bold text-[var(--muted)]">Retrieved chunks<input className="control mt-1" type="number" min="1" max="20" value={topK} onChange={(event) => setTopK(Number(event.target.value))} /></label>
              <div className="mt-4 divide-y divide-[var(--line)]">
                <Toggle checked={useRagas} onChange={setUseRagas} label="Ragas LLM evaluation" description="Add context precision and recall using Claude." />
                <Toggle checked={persistEmbeddings} onChange={setPersistEmbeddings} label="Keep embeddings for export" description="Retain vectors after the run for NPZ, FAISS, and JSONL exports." />
                <Toggle checked={useVectorDb} onChange={setUseVectorDb} label="Write to ChromaDB" description="Create temporary cosine-indexed vector collections." />
              </div>
            </section>
            {error && <div className="rounded-md border border-[var(--coral)] bg-[var(--soft-coral)] p-4 text-sm font-semibold text-[var(--coral-dark)]">{error}</div>}
            <button disabled={submitting} className="btn-primary w-full disabled:cursor-wait disabled:opacity-70" type="submit">{submitting ? <><LoaderCircle className="animate-spin" size={17} /> Creating run</> : <>Run {chunkers.length * models.length} configurations <ArrowRight size={17} /></>}</button>
          </aside>
        </div>
      </form>
    </main>
  );
}