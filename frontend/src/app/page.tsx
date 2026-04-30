import Link from "next/link";

export default function HomePage() {
  return (
    <main className="min-h-screen bg-neutral-950 text-neutral-50 flex flex-col items-center justify-center">
      <div className="max-w-2xl w-full px-6 text-center space-y-6">
        <h1 className="text-4xl md:text-5xl font-semibold tracking-tight">
          protoRAG
        </h1>
        <p className="text-neutral-400">
          Local-first, privacy-preserving RAG workspace with dynamic LLM configuration.
        </p>
        <div className="flex items-center justify-center gap-4 mt-6">
          <Link
            href="/chat"
            className="rounded-full bg-neutral-100 text-neutral-900 px-6 py-2 text-sm font-medium hover:bg-white transition-colors"
          >
            Open chat
          </Link>
          <Link
            href="/documents"
            className="rounded-full border border-neutral-700 px-6 py-2 text-sm font-medium text-neutral-200 hover:border-neutral-500 transition-colors"
          >
            Manage documents
          </Link>
        </div>
      </div>
    </main>
  );
}
