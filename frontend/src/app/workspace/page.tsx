import { fetchConfig } from "@/lib/api";
import { BenchmarkForm } from "@/components/benchmark-form";

export default async function WorkspacePage() {
  let config;
  try {
    config = await fetchConfig();
  } catch {
    return (
      <main className="mx-auto max-w-7xl px-5 py-16 md:px-10">
        <p className="eyebrow text-[var(--coral)]">Service unavailable</p>
        <h1 className="mt-3 text-4xl font-bold">Start the Python API to configure a benchmark.</h1>
        <p className="mt-4 text-[var(--muted)]">Run <code className="font-mono">uvicorn api:app --reload</code> from the RAG-VAL directory.</p>
      </main>
    );
  }

  return <BenchmarkForm config={config} />;
}