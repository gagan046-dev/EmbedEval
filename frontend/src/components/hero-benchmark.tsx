"use client";

import { Bar, BarChart, CartesianGrid, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

const data = [
  { name: "Recursive + BGE", f1: 0.91, rougeL: 0.86 },
  { name: "Semantic + E5", f1: 0.87, rougeL: 0.82 },
  { name: "Window + MiniLM", f1: 0.79, rougeL: 0.75 },
  { name: "Fixed + OpenAI", f1: 0.84, rougeL: 0.8 },
];

export function HeroBenchmark() {
  return (
    <div className="overflow-hidden rounded-lg border border-[var(--line)] bg-[var(--surface)] text-[var(--ink)] shadow-[14px_14px_0_#d7d3c8]">
      <div className="flex items-center justify-between border-b border-[var(--line)] px-4 py-3 md:px-5 md:py-4">
        <div>
          <p className="eyebrow text-[var(--muted)]">Live comparison</p>
          <p className="mt-1 font-bold">Retrieval quality by configuration</p>
        </div>
        <span className="rounded-full bg-[var(--teal)] px-3 py-1 text-xs font-bold text-white">Run complete</span>
      </div>
      <div className="grid grid-cols-3 border-b border-[var(--line)]">
        {[['Best F1', '0.91'], ['ROUGE-L', '0.86'], ['Configs', '12']].map(([label, value]) => (
          <div key={label} className="border-r border-[var(--line)] px-3 py-2.5 last:border-r-0 md:px-4 md:py-4">
            <p className="eyebrow text-[var(--muted)]">{label}</p>
            <p className="metric-number mt-1 text-xl font-semibold md:text-2xl">{value}</p>
          </div>
        ))}
      </div>
      <div className="space-y-2 p-3 sm:hidden">
        {data.slice(0, 3).map((item) => (
          <div key={item.name} className="grid grid-cols-[105px_1fr_26px] items-center gap-2 text-[10px]">
            <span className="truncate text-[var(--muted)]">{item.name}</span>
            <span className="h-2 overflow-hidden rounded-full bg-[var(--line)]"><span className="block h-full rounded-full bg-[var(--coral)]" style={{ width: `${item.f1 * 100}%` }} /></span>
            <strong className="metric-number">{item.f1.toFixed(2)}</strong>
          </div>
        ))}
      </div>
      <div className="hidden h-[225px] p-3 sm:block lg:h-[240px]">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={data} layout="vertical" margin={{ left: 12, right: 12, top: 8 }}>
            <CartesianGrid horizontal={false} stroke="#e1e0d9" />
            <XAxis type="number" domain={[0, 1]} hide />
            <YAxis dataKey="name" type="category" width={118} tick={{ fill: "#171916", fontSize: 11 }} axisLine={false} tickLine={false} />
            <Tooltip cursor={{ fill: "rgba(255,255,255,.04)" }} contentStyle={{ background: "#fffefa", border: 0, borderRadius: 6, color: "#171916" }} />
            <Legend verticalAlign="top" height={30} formatter={(value) => <span className="text-[var(--ink)]">{value}</span>} />
            <Bar dataKey="f1" name="F1" fill="#e65336" radius={[0, 3, 3, 0]} />
            <Bar dataKey="rougeL" name="ROUGE-L" fill="#44a99d" radius={[0, 3, 3, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}