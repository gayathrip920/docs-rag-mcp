import type { Metadata } from "next";
import Link from "next/link";
import "./globals.css";

export const metadata: Metadata = {
  title: "Docs search",
  description: "Search the FastAPI docs by meaning, with links back to the source pages.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="font-sans antialiased">
        <header className="border-b border-slate-200 dark:border-slate-800">
          <nav className="mx-auto flex max-w-3xl items-center justify-between px-4 py-3">
            <Link href="/" className="font-semibold">
              Docs search
            </Link>
            <Link href="/metrics" className="text-sm text-slate-600 hover:underline dark:text-slate-400">
              Metrics
            </Link>
          </nav>
        </header>
        <main className="mx-auto max-w-3xl px-4 py-8">{children}</main>
        <footer className="mx-auto max-w-3xl px-4 pb-10 text-xs text-slate-500">
          Searches the FastAPI documentation (MIT license). Questions are saved to measure speed.
          Your API key is never saved.
        </footer>
      </body>
    </html>
  );
}
