import { AgentReport } from "@/components/agent-report";

export default async function ReportPage({ params }: PageProps<"/reports/[runId]">) {
  const { runId } = await params;
  return <AgentReport runId={runId} />;
}