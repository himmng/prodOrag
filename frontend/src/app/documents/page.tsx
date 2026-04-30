"use client";

import { FormEvent, useEffect, useState } from "react";

type Document = {
  id: string;
  workspace_id: string;
  filename: string;
  mime_type: string;
  size_bytes: number;
  status: string;
  num_chunks: number;
  created_at: string;
};

async function api<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${process.env.NEXT_PUBLIC_BACKEND_URL ?? "http://localhost:8000"}${path}`, {
    ...options,
    headers: {
      "Content-Type": "application/json",
      ...(options?.headers ?? {}),
    },
  });

  if (!res.ok) {
    throw new Error(`Request failed: ${res.status}`);
  }

  return res.json();
}

export default function DocumentsPage() {
  const [documents, setDocuments] = useState<Document[]>([]);
  const [loading, setLoading] = useState(true);
  const [creating, setCreating] = useState(false);
  const [filename, setFilename] = useState("");
  const [mimeType, setMimeType] = useState("text/plain");
  const [sizeBytes, setSizeBytes] = useState("0");

  useEffect(() => {
    let cancelled = false;

    async function load() {
      try {
        const docs = await api<Document[]>("/documents");
        if (!cancelled) setDocuments(docs);
      } catch (e) {
        console.error(e);
      } finally {
        if (!cancelled) setLoading(false);
      }
    }

    load();

    return () => {
      cancelled = true;
    };
  }, []);

  async function handleCreate(e: FormEvent) {
    e.preventDefault();
    if (!filename.trim()) return;

    setCreating(true);

    try {
      const created = await api<Document>("/documents", {
        method: "POST",
        body: JSON.stringify({
          workspace_id: "00000000-0000-0000-0000-000000000000",
          filename: filename.trim(),
          mime_type: mimeType,
          size_bytes: Number(sizeBytes) || 0,
        }),
      });
      setDocuments((prev) => [created, ...prev]);
      setFilename("");
      setSizeBytes("0");
    } catch (e) {
      console.error(e);
    } finally {
      setCreating(false);
    }
  }

  async function handleDelete(id: string) {
    const existing = documents.find((d) => d.id === id);
    if (!existing) return;

    setDocuments((prev) => prev.filter((d) => d.id !== id));

    try {
      await api<void>(`/documents/${id}`, { method: "DELETE" });
    } catch (e) {
      console.error(e);
      setDocuments((prev) => [...prev, existing]);
    }
  }

  return (
    <main className="min-h-screen bg-neutral-950 text-neutral-50 flex flex-col">
      <header className="border-b border-neutral-900 px-4 py-3 flex items-center justify-between">
        <h1 className="text-sm font-medium text-neutral-200">Documents</h1>
      </header>
      <div className="max-w-4xl w-full mx-auto px-4 py-4 space-y-4">
        <section className="rounded-lg border border-neutral-900 bg-neutral-950/60 p-4">
          <h2 className="text-xs font-semibold text-neutral-400 mb-3 uppercase tracking-wide">
            Add document (stub)
          </h2>
          <form onSubmit={handleCreate} className="flex flex-wrap gap-3 items-center">
            <input
              className="flex-1 min-w-[140px] bg-neutral-900 border border-neutral-800 rounded-md px-3 py-2 text-sm outline-none focus:border-neutral-500"
              placeholder="Filename"
              value={filename}
              onChange={(e) => setFilename(e.target.value)}
            />
            <input
              className="w-32 bg-neutral-900 border border-neutral-800 rounded-md px-3 py-2 text-sm outline-none focus:border-neutral-500"
              placeholder="MIME type"
              value={mimeType}
              onChange={(e) => setMimeType(e.target.value)}
            />
            <input
              className="w-32 bg-neutral-900 border border-neutral-800 rounded-md px-3 py-2 text-sm outline-none focus:border-neutral-500"
              placeholder="Size (bytes)"
              value={sizeBytes}
              onChange={(e) => setSizeBytes(e.target.value)}
            />
            <button
              type="submit"
              disabled={creating || !filename.trim()}
              className="text-xs font-medium px-4 py-2 rounded-full bg-neutral-100 text-neutral-900 disabled:opacity-40"
            >
              {creating ? "Adding" : "Add"}
            </button>
          </form>
          <p className="mt-2 text-[11px] text-neutral-600">
            This is a stub that creates document metadata rows; real file upload + chunking into Qdrant can be wired later.
          </p>
        </section>

        <section className="rounded-lg border border-neutral-900 bg-neutral-950/60 p-4">
          <h2 className="text-xs font-semibold text-neutral-400 mb-3 uppercase tracking-wide">
            Documents
          </h2>
          {loading ? (
            <div className="text-neutral-500 text-sm">Loading documents...</div>
          ) : documents.length === 0 ? (
            <div className="text-neutral-600 text-sm">No documents yet.</div>
          ) : (
            <div className="divide-y divide-neutral-900 text-sm">
              {documents.map((doc) => (
                <div key={doc.id} className="flex items-center justify-between py-2">
                  <div className="space-y-1">
                    <div className="font-medium text-neutral-100">{doc.filename}</div>
                    <div className="text-[11px] text-neutral-500">
                      {doc.mime_type} · {doc.size_bytes} bytes · status {doc.status} · chunks {" "}
                      {doc.num_chunks}
                    </div>
                  </div>
                  <button
                    onClick={() => handleDelete(doc.id)}
                    className="text-[11px] text-red-400 hover:text-red-300"
                  >
                    Delete
                  </button>
                </div>
              ))}
            </div>
          )}
        </section>
      </div>
    </main>
  );
}
