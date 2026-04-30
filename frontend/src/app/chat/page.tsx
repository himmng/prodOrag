"use client";

import { useEffect, useState } from "react";
import dynamic from "next/dynamic";

const MotionDiv = dynamic(
  () => import("framer-motion").then((mod) => mod.motion.div),
  { ssr: false }
);

type ChatMessage = {
  id: string;
  role: "user" | "assistant";
  content: string;
  created_at: string;
};

type ChatSession = {
  id: string;
  title: string;
  created_at: string;
};

type ChatSessionWithMessages = ChatSession & {
  messages: ChatMessage[];
};

type DocumentItem = {
  id: string;
  filename: string;
  mime_type: string;
  size_bytes: number;
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

export default function ChatPage() {
  const [session, setSession] = useState<ChatSessionWithMessages | null>(null);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [initializing, setInitializing] = useState(true);
  const [documents, setDocuments] = useState<DocumentItem[]>([]);
  const [uploading, setUploading] = useState(false);

  useEffect(() => {
    let cancelled = false;

    async function ensureSession() {
      try {
        const created = await api<ChatSession>("/chat/sessions", {
          method: "POST",
          body: JSON.stringify({ title: "New chat" }),
        });

        if (cancelled) return;

        const full = await api<ChatSessionWithMessages>(`/chat/sessions/${created.id}`);
        if (!cancelled) {
          setSession(full);
          try {
            const docsRes = await fetch(
              `${process.env.NEXT_PUBLIC_BACKEND_URL ?? "http://localhost:8000"}/documents/by-session/${created.id}`
            );
            if (docsRes.ok) {
              const docsJson = (await docsRes.json()) as DocumentItem[];
              setDocuments(docsJson);
            }
          } catch (err) {
            console.error(err);
          }
        }
      } catch (e) {
        console.error(e);
      } finally {
        if (!cancelled) setInitializing(false);
      }
    }

    ensureSession();

    return () => {
      cancelled = true;
    };
  }, []);

  const hasDocuments = documents.length > 0;
  const hasConversation = (session?.messages.length ?? 0) > 0;
  const canExport = hasDocuments && hasConversation;

  async function handleSend() {
    if (!session || !input.trim()) return;
    setLoading(true);
    const content = input.trim();
    setInput("");

    try {
      const created = await api<ChatMessage>(`/chat/sessions/${session.id}/messages`, {
        method: "POST",
        body: JSON.stringify({ content }),
      });

      setSession({
        ...session,
        messages: [...session.messages, created],
      });
    } catch (e) {
      console.error(e);
      setInput(content);
    } finally {
      setLoading(false);
    }
  }

  async function handleUpload(files: FileList | null) {
    if (!session || !files || files.length === 0) return;
    setUploading(true);
    try {
      const backendUrl = process.env.NEXT_PUBLIC_BACKEND_URL ?? "http://localhost:8000";
      const formData = new FormData();
      Array.from(files).forEach((file) => {
        formData.append("files", file);
      });
      const res = await fetch(`${backendUrl}/documents/upload/${session.id}`, {
        method: "POST",
        body: formData,
      });
      if (res.ok) {
        const uploaded = (await res.json()) as DocumentItem[];
        setDocuments((prev) => [...prev, ...uploaded]);
      }
    } catch (err) {
      console.error(err);
    } finally {
      setUploading(false);
    }
  }

  async function handleExportSummary() {
    if (!session) return;
    try {
      const backendUrl = process.env.NEXT_PUBLIC_BACKEND_URL ?? "http://localhost:8000";
      const res = await fetch(`${backendUrl}/chat/sessions/${session.id}/export-summary`);
      if (!res.ok) return;
      const data = await res.json();
      const blob = new Blob([JSON.stringify(data, null, 2)], {
        type: "application/json",
      });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `chat-summary-${session.id}.json`;
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(url);
    } catch (err) {
      console.error(err);
    }
  }

  async function handleExportIRAG() {
    if (!session) return;
    try {
      const backendUrl = process.env.NEXT_PUBLIC_BACKEND_URL ?? "http://localhost:8000";
      const res = await fetch(`${backendUrl}/chat/sessions/${session.id}/export-irag`);
      if (!res.ok) return;
      const data = await res.json();
      const blob = new Blob([JSON.stringify(data, null, 2)], {
        type: "application/json",
      });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `chat-irag-${session.id}.json`;
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(url);
    } catch (err) {
      console.error(err);
    }
  }

  return (
    <main className="min-h-screen bg-neutral-50 text-neutral-900 dark:bg-neutral-950 dark:text-neutral-50 flex flex-col">
      <header className="border-b border-neutral-200 dark:border-neutral-900 px-4 py-3 flex flex-col gap-2">
        <div className="flex items-center justify-between">
          <h1 className="text-sm font-medium text-neutral-800 dark:text-neutral-200">Chat</h1>
          <div className="text-xs text-neutral-500 dark:text-neutral-500">
            Session: {session?.title ?? "initializing..."}
          </div>
        </div>
        <div className="flex flex-wrap items-center justify-between gap-2 text-xs">
          <div className="flex items-center gap-2">
            <label className="inline-flex items-center gap-2 rounded-full border border-dashed border-neutral-300 px-3 py-1 text-neutral-600 hover:border-neutral-400 dark:border-neutral-700 dark:text-neutral-300 dark:hover:border-neutral-500 cursor-pointer">
              <input
                type="file"
                multiple
                className="hidden"
                onChange={(e) => handleUpload(e.target.files)}
              />
              <span className="inline-flex h-5 w-5 items-center justify-center rounded-full bg-neutral-100 text-neutral-700 dark:bg-neutral-800 dark:text-neutral-100 text-[10px]">
                6
              </span>
              <span>{uploading ? "Uploading..." : "Upload documents"}</span>
            </label>
            {documents.length > 0 && (
              <div className="flex flex-wrap gap-1 max-w-xs">
                {documents.map((doc) => (
                  <span
                    key={doc.id}
                    className="inline-flex items-center rounded-full bg-neutral-100 px-2 py-0.5 text-[10px] text-neutral-700 dark:bg-neutral-800 dark:text-neutral-200"
                  >
                    {doc.filename}
                  </span>
                ))}
              </div>
            )}
          </div>
          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={handleExportSummary}
              disabled={!canExport}
              className="inline-flex items-center gap-1 rounded-full border px-3 py-1 text-[11px] text-neutral-700 hover:bg-neutral-100 disabled:opacity-40 dark:border-neutral-700 dark:text-neutral-200 dark:hover:bg-neutral-800"
            >
              <span>Export chat summary</span>
              <span className="inline-flex h-4 w-4 items-center justify-center rounded-full border border-neutral-300 text-[9px] text-neutral-500 dark:border-neutral-600 dark:text-neutral-400">
                i
              </span>
            </button>
            <button
              type="button"
              onClick={handleExportIRAG}
              disabled={!canExport}
              className="inline-flex items-center gap-1 rounded-full border px-3 py-1 text-[11px] text-neutral-700 hover:bg-neutral-100 disabled:opacity-40 dark:border-neutral-700 dark:text-neutral-200 dark:hover:bg-neutral-800"
            >
              <span>Export iRAG</span>
              <span className="inline-flex h-4 w-4 items-center justify-center rounded-full border border-neutral-300 text-[9px] text-neutral-500 dark:border-neutral-600 dark:text-neutral-400">
                i
              </span>
            </button>
          </div>
        </div>
      </header>
      <MotionDiv
        initial={{ opacity: 0, y: 8 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.3, ease: "easeOut" }}
        className="flex-1 flex flex-col max-w-3xl w-full mx-auto px-4 py-4 gap-4"
      >
        <div className="flex-1 overflow-y-auto rounded-lg border border-neutral-200 bg-white/80 p-4 space-y-3 dark:border-neutral-900 dark:bg-neutral-950/60">
          {initializing && <div className="text-neutral-500 text-sm">Creating session...</div>}
          {!initializing && session && session.messages.length === 0 && (
            <div className="text-neutral-500 text-sm">Send a message to get started.</div>
          )}
          {!initializing &&
            session?.messages.map((m) => (
              <div
                key={m.id}
                className={`flex ${m.role === "user" ? "justify-end" : "justify-start"}`}
              >
                <div
                  className={`max-w-[80%] rounded-2xl px-3 py-2 text-sm leading-relaxed ${{
                    user: "bg-neutral-900 text-neutral-50 dark:bg-neutral-100 dark:text-neutral-900",
                    assistant: "bg-neutral-100 text-neutral-900 dark:bg-neutral-900 dark:text-neutral-100",
                  }[m.role]}`}
                >
                  {m.content}
                </div>
              </div>
            ))}
        </div>
        <div className="border border-neutral-200 rounded-full flex items-center px-3 py-2 gap-2 bg-white dark:border-neutral-900 dark:bg-neutral-950">
          <input
            className="flex-1 bg-transparent outline-none text-sm placeholder:text-neutral-400 dark:placeholder:text-neutral-600"
            placeholder="Ask a question..."
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                handleSend();
              }
            }}
            disabled={loading || initializing}
          />
          <button
            onClick={handleSend}
            disabled={loading || initializing || !input.trim()}
            className="text-xs font-medium px-3 py-1 rounded-full bg-neutral-900 text-neutral-50 disabled:opacity-40 dark:bg-neutral-100 dark:text-neutral-900"
          >
            {loading ? "Sending" : "Send"}
          </button>
        </div>
      </MotionDiv>
    </main>
  );
}

