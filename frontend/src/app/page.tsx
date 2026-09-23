import Link from "next/link";
import { ArrowRight, Database, Gauge, ScanSearch } from "lucide-react";
import { HeroBenchmark } from "@/components/hero-benchmark";

const stages = [
  { icon: ScanSearch, label: "Prepare", text: "Parse source documents and build a grounded evaluation set." },
  { icon: Database, label: "Benchmark", text: "Cross-test chunking strategies and embedding models in parallel." },
  { icon: Gauge, label: "Decide", text: "Compare quality, latency, cost, and agent recommendations." },
];

export default function Home() {
  return (
    <main>
      <section className="page-grid border-b border-[var(--line)] px-5 pb-5 pt-6 md:min-h-[calc(100vh-150px)] md:px-10 md:pb-8 md:pt-14">
        <div className="mx-auto grid max-w-7xl items-center gap-6 lg:grid-cols-[0.9fr_1.1fr] lg:gap-12">
          <div className="rise max-w-2xl">
            <p className="eyebrow mb-3 text-[var(--coral)] md:mb-5">RAG evaluation workbench</p>
            <h1 className="text-5xl font-extrabold leading-[0.98] sm:text-6xl lg:text-7xl">
              EmbedEval AI
            </h1>
            <p className="mt-4 max-w-xl text-base leading-6 text-[var(--muted)] md:mt-7 md:text-lg md:leading-8">
              Replace retrieval guesswork with measured evidence. Compare chunking,
              embeddings, relevance, latency, and cost in one reproducible run.
            </p>
            <div className="mt-5 flex flex-wrap gap-3 md:mt-9">
              <Link className="btn-primary" href="/workspace">
                Start a benchmark <ArrowRight size={17} />
              </Link>
              <a className="btn-secondary" href="#method">
                See the method
              </a>
            </div>
            <div className="mt-12 hidden gap-8 border-t border-[var(--line)] pt-5 text-sm sm:flex">
              <div><strong className="metric-number block text-xl">11</strong><span className="text-[var(--muted)]">chunking methods</span></div>
              <div><strong className="metric-number block text-xl">9+</strong><span className="text-[var(--muted)]">embedding models</span></div>
              <div><strong className="metric-number block text-xl">6</strong><span className="text-[var(--muted)]">quality signals</span></div>
            </div>
          </div>
          <div className="rise rise-delay-1 min-w-0">
            <HeroBenchmark />
          </div>
        </div>
        <div className="mx-auto mt-5 max-w-7xl text-center md:mt-10">
          <span className="eyebrow text-[var(--muted)]">The workflow continues below</span>
        </div>
      </section>

      <section id="method" className="bg-[var(--surface)] px-5 py-20 md:px-10">
        <div className="mx-auto max-w-7xl">
          <div className="grid gap-8 border-b border-[var(--line)] pb-10 md:grid-cols-[0.8fr_1.2fr]">
            <p className="eyebrow text-[var(--teal)]">Evaluation method</p>
            <h2 className="max-w-3xl text-3xl font-bold leading-tight md:text-4xl">
              A decision system for retrieval architecture, not another opaque scorecard.
            </h2>
          </div>
          <div className="grid md:grid-cols-3">
            {stages.map(({ icon: Icon, label, text }, index) => (
              <article key={label} className="border-b border-[var(--line)] py-8 md:border-b-0 md:border-r md:px-8 md:first:pl-0 md:last:border-r-0">
                <div className="mb-8 flex items-center justify-between">
                  <Icon size={22} strokeWidth={1.8} className="text-[var(--coral)]" />
                  <span className="metric-number text-xs text-[var(--muted)]">0{index + 1}</span>
                </div>
                <h3 className="text-xl font-bold">{label}</h3>
                <p className="mt-3 leading-7 text-[var(--muted)]">{text}</p>
              </article>
            ))}
          </div>
        </div>
      </section>
    </main>
  );
}
