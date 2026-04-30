import type { Metadata } from "next";
import localFont from "next/font/local";
import Link from "next/link";
import "./globals.css";
import { ThemeProvider } from "@/components/theme-provider";
import { ThemeToggle } from "@/components/theme-toggle";

const geistSans = localFont({
  src: "./fonts/GeistVF.woff",
  variable: "--font-geist-sans",
  weight: "100 900",
});
const geistMono = localFont({
  src: "./fonts/GeistMonoVF.woff",
  variable: "--font-geist-mono",
  weight: "100 900",
});

export const metadata: Metadata = {
  title: "protoRAG",
  description: "Local-first RAG prototype",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased bg-white text-neutral-900 dark:bg-neutral-950 dark:text-neutral-50`}
      >
        <ThemeProvider>
          <div className="min-h-screen flex flex-col">
            <header className="border-b border-neutral-200 bg-white/80 text-neutral-800 dark:border-neutral-900 dark:bg-neutral-950/80 dark:text-neutral-100 backdrop-blur flex items-center justify-between px-4 py-3">
              <div className="flex items-center gap-3">
                <Link href="/" className="text-sm font-semibold tracking-tight">
                  protoRAG
                </Link>
              </div>
              <div className="flex items-center gap-4 text-xs text-neutral-500 dark:text-neutral-400">
                <nav className="flex items-center gap-4">
                  <Link
                    href="/chat"
                    className="hover:text-neutral-900 dark:hover:text-neutral-100 transition-colors"
                  >
                    Chat
                  </Link>
                  <Link
                    href="/documents"
                    className="hover:text-neutral-900 dark:hover:text-neutral-100 transition-colors"
                  >
                    Documents
                  </Link>
                  <Link
                    href="/settings"
                    className="hover:text-neutral-900 dark:hover:text-neutral-100 transition-colors"
                  >
                    Settings
                  </Link>
                </nav>
                <ThemeToggle />
              </div>
            </header>
            <div className="flex-1 flex flex-col">{children}</div>
          </div>
        </ThemeProvider>
      </body>
    </html>
  );
}
