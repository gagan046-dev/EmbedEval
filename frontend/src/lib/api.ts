import type { AppConfig, BenchmarkRun } from "@/lib/types";

export const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

async function parseResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const body = await response.json().catch(() => ({ detail: "Request failed" }));
    throw new Error(body.detail ?? "Request failed");
  }
  return response.json() as Promise<T>;
}

export async function fetchConfig(): Promise<AppConfig> {
  return parseResponse(await fetch(`${API_URL}/api/config`, { cache: "no-store" }));
}

export async function createRun(formData: FormData): Promise<BenchmarkRun> {
  return parseResponse(await fetch(`${API_URL}/api/runs`, { method: "POST", body: formData }));
}

export async function fetchRun(runId: string): Promise<BenchmarkRun> {
  return parseResponse(await fetch(`${API_URL}/api/runs/${runId}`, { cache: "no-store" }));
}

export async function generateReport(runId: string, anthropicKey: string): Promise<string> {
  const data = await parseResponse<{ report: string }>(await fetch(`${API_URL}/api/runs/${runId}/report`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ anthropic_key: anthropicKey }),
  }));
  return data.report;
}

export function exportUrl(runId: string, format: "json" | "csv") {
  return `${API_URL}/api/runs/${runId}/exports/${format}`;
}

export function embeddingExportUrl(runId: string, configKey: string, format: "npz" | "faiss" | "jsonl") {
  const params = new URLSearchParams({ config_key: configKey, format_name: format });
  return `${API_URL}/api/runs/${runId}/embeddings?${params}`;
}