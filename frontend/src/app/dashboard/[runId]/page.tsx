import { RunDashboard } from "@/components/run-dashboard";

export default async function DashboardPage({ params }: PageProps<"/dashboard/[runId]">) {
  const { runId } = await params;
  return <RunDashboard runId={runId} />;
}