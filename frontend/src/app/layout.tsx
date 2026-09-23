import type { Metadata } from "next";
import { IBM_Plex_Mono, Manrope } from "next/font/google";
import { SiteHeader } from "@/components/site-header";
import "./globals.css";

const manrope = Manrope({ subsets: ["latin"], variable: "--font-sans" });
const plexMono = IBM_Plex_Mono({
  subsets: ["latin"],
  weight: ["400", "500", "600"],
  variable: "--font-mono",
});

export const metadata: Metadata = {
  title: "EmbedEval AI | RAG Evaluation Workbench",
  description: "Benchmark retrieval strategies with evidence, not instinct.",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html lang="en" data-scroll-behavior="smooth">
      <body className={`${manrope.variable} ${plexMono.variable}`}>
        <SiteHeader />
        {children}
      </body>
    </html>
  );
}
