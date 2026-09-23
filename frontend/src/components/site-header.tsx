"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { Activity, ArrowUpRight } from "lucide-react";

const links = [
  { href: "/", label: "Home" },
  { href: "/workspace", label: "Workspace" },
];

export function SiteHeader() {
  const pathname = usePathname();

  return (
    <header className="sticky top-0 z-50 border-b border-[var(--line)] bg-[rgba(245,243,237,0.94)] backdrop-blur-md">
      <div className="mx-auto flex h-[68px] max-w-7xl items-center justify-between px-5 md:px-10">
        <Link href="/" className="flex items-center gap-2 font-extrabold">
          <span className="flex size-8 items-center justify-center rounded-md bg-[var(--ink)] text-white"><Activity size={18} /></span>
          EmbedEval <span className="text-[var(--coral)]">AI</span>
        </Link>
        <nav className="flex items-center gap-1" aria-label="Main navigation">
          {links.map((link) => (
            <Link
              key={link.href}
              href={link.href}
              className={`rounded-md px-3 py-2 text-sm font-semibold ${pathname === link.href ? "bg-[var(--ink)] text-white" : "text-[var(--muted)] hover:text-[var(--ink)]"}`}
            >
              {link.label}
            </Link>
          ))}
          <Link href="/workspace" className="ml-2 hidden items-center gap-1 text-sm font-bold text-[var(--coral)] sm:flex">
            New run <ArrowUpRight size={15} />
          </Link>
        </nav>
      </div>
    </header>
  );
}