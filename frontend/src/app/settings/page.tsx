"use client";

import { FormEvent, useEffect, useState } from "react";

type ConfigProfile = {
  id: string;
  name: string;
  is_default: boolean;
  llm_provider: string;
  llm_base_url: string;
  llm_model_id: string;
  llm_api_key: string | null;
  embedding_provider: string;
  embedding_base_url: string;
  embedding_model_name: string;
  embedding_api_key: string | null;
  retrieval_top_k: number;
  retrieval_score_threshold: number | null;
  hybrid_search_enabled: boolean;
  reranker_model: string | null;
  created_at: string;
  updated_at: string;
};

type ConfigProfilePayload = {
  name: string;
  is_default: boolean;
  llm_provider: string;
  llm_base_url: string;
  llm_model_id: string;
  llm_api_key: string | null;
  embedding_provider: string;
  embedding_base_url: string;
  embedding_model_name: string;
  embedding_api_key: string | null;
  retrieval_top_k: number;
  retrieval_score_threshold: number | null;
  hybrid_search_enabled: boolean;
  reranker_model: string | null;
};

type ConfigProfileFormState = {
  name: string;
  is_default: boolean;
  llm_provider: string;
  llm_base_url: string;
  llm_model_id: string;
  llm_api_key: string;
  embedding_provider: string;
  embedding_base_url: string;
  embedding_model_name: string;
  embedding_api_key: string;
  retrieval_top_k: string;
  retrieval_score_threshold: string;
  hybrid_search_enabled: boolean;
  reranker_model: string;
};

type ValidationErrors = Record<string, string>;

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

function profileToForm(profile: ConfigProfile): ConfigProfileFormState {
  return {
    name: profile.name,
    is_default: profile.is_default,
    llm_provider: profile.llm_provider,
    llm_base_url: profile.llm_base_url,
    llm_model_id: profile.llm_model_id,
    llm_api_key: profile.llm_api_key ?? "",
    embedding_provider: profile.embedding_provider,
    embedding_base_url: profile.embedding_base_url,
    embedding_model_name: profile.embedding_model_name,
    embedding_api_key: profile.embedding_api_key ?? "",
    retrieval_top_k: String(profile.retrieval_top_k),
    retrieval_score_threshold:
      profile.retrieval_score_threshold === null
        ? ""
        : String(profile.retrieval_score_threshold),
    hybrid_search_enabled: profile.hybrid_search_enabled,
    reranker_model: profile.reranker_model ?? "",
  };
}

function defaultFormState(): ConfigProfileFormState {
  return {
    name: "local-default",
    is_default: true,
    llm_provider: "ollama",
    llm_base_url: "http://localhost:11434",
    llm_model_id: "llama3",
    llm_api_key: "",
    embedding_provider: "local",
    embedding_base_url: "http://localhost:11434",
    embedding_model_name: "nomic-embed-text",
    embedding_api_key: "",
    retrieval_top_k: "8",
    retrieval_score_threshold: "",
    hybrid_search_enabled: false,
    reranker_model: "",
  };
}

function formToPayload(form: ConfigProfileFormState): ConfigProfilePayload {
  const retrievalTopK = Number(form.retrieval_top_k);
  const retrievalScoreThreshold = form.retrieval_score_threshold.trim();

  return {
    name: form.name.trim(),
    is_default: form.is_default,
    llm_provider: form.llm_provider.trim(),
    llm_base_url: form.llm_base_url.trim(),
    llm_model_id: form.llm_model_id.trim(),
    llm_api_key: form.llm_api_key.trim() === "" ? null : form.llm_api_key.trim(),
    embedding_provider: form.embedding_provider.trim(),
    embedding_base_url: form.embedding_base_url.trim(),
    embedding_model_name: form.embedding_model_name.trim(),
    embedding_api_key:
      form.embedding_api_key.trim() === "" ? null : form.embedding_api_key.trim(),
    retrieval_top_k: Number.isFinite(retrievalTopK) && retrievalTopK > 0 ? retrievalTopK : 8,
    retrieval_score_threshold:
      retrievalScoreThreshold === ""
        ? null
        : Number.isFinite(Number(retrievalScoreThreshold))
        ? Number(retrievalScoreThreshold)
        : null,
    hybrid_search_enabled: form.hybrid_search_enabled,
    reranker_model: form.reranker_model.trim() === "" ? null : form.reranker_model.trim(),
  };
}

export default function SettingsPage() {
  const [form, setForm] = useState<ConfigProfileFormState>(defaultFormState);
  const [initialForm, setInitialForm] = useState<ConfigProfileFormState | null>(null);
  const [profileId, setProfileId] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [errors, setErrors] = useState<ValidationErrors>({});
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [successMessage, setSuccessMessage] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;

    async function load() {
      try {
        const profiles = await api<ConfigProfile[]>("/config/profiles");
        if (cancelled) return;
        if (profiles.length === 0) {
          const defaults = defaultFormState();
          setForm(defaults);
          setInitialForm(defaults);
          setProfileId(null);
        } else {
          const active = profiles.find((p) => p.is_default) ?? profiles[0];
          const mapped = profileToForm(active);
          setForm(mapped);
          setInitialForm(mapped);
          setProfileId(active.id);
        }
      } catch {
        if (!cancelled) {
          setErrorMessage("Failed to load configuration.");
        }
      } finally {
        if (!cancelled) setLoading(false);
      }
    }

    load();

    return () => {
      cancelled = true;
    };
  }, []);

  function updateField<K extends keyof ConfigProfileFormState>(key: K, value: ConfigProfileFormState[K]) {
    setForm((prev) => ({ ...prev, [key]: value }));
  }

  function validate(): boolean {
    const nextErrors: ValidationErrors = {};

    if (!form.name.trim()) {
      nextErrors.name = "Name is required";
    }
    if (!form.llm_provider.trim()) {
      nextErrors.llm_provider = "LLM provider is required";
    }
    if (!form.llm_model_id.trim()) {
      nextErrors.llm_model_id = "LLM model is required";
    }
    if (!form.embedding_provider.trim()) {
      nextErrors.embedding_provider = "Embedding provider is required";
    }
    if (!form.embedding_model_name.trim()) {
      nextErrors.embedding_model_name = "Embedding model is required";
    }
    if (!form.llm_base_url.trim()) {
      nextErrors.llm_base_url = "LLM base URL is required";
    } else {
      try {
        new URL(form.llm_base_url.trim());
      } catch {
        nextErrors.llm_base_url = "LLM base URL must be a valid URL";
      }
    }
    if (!form.embedding_base_url.trim()) {
      nextErrors.embedding_base_url = "Embedding base URL is required";
    } else {
      try {
        new URL(form.embedding_base_url.trim());
      } catch {
        nextErrors.embedding_base_url = "Embedding base URL must be a valid URL";
      }
    }

    const retrievalTopK = Number(form.retrieval_top_k);
    if (!Number.isFinite(retrievalTopK) || retrievalTopK <= 0) {
      nextErrors.retrieval_top_k = "Top K must be a positive number";
    }

    const retrievalScoreThreshold = form.retrieval_score_threshold.trim();
    if (retrievalScoreThreshold !== "") {
      const n = Number(retrievalScoreThreshold);
      if (!Number.isFinite(n)) {
        nextErrors.retrieval_score_threshold = "Score threshold must be a number";
      }
    }

    setErrors(nextErrors);

    return Object.keys(nextErrors).length === 0;
  }

  async function handleSubmit(e: FormEvent) {
    e.preventDefault();
    setErrorMessage(null);
    setSuccessMessage(null);

    if (!validate()) {
      return;
    }

    setSaving(true);

    try {
      const payload = formToPayload(form);
      if (profileId === null) {
        const created = await api<ConfigProfile>("/config/profiles", {
          method: "POST",
          body: JSON.stringify(payload),
        });
        const mapped = profileToForm(created);
        setForm(mapped);
        setInitialForm(mapped);
        setProfileId(created.id);
      } else {
        const updated = await api<ConfigProfile>(`/config/profiles/${profileId}`, {
          method: "PUT",
          body: JSON.stringify(payload),
        });
        const mapped = profileToForm(updated);
        setForm(mapped);
        setInitialForm(mapped);
      }
      setSuccessMessage("Settings saved.");
    } catch {
      setErrorMessage("Failed to save settings.");
    } finally {
      setSaving(false);
    }
  }

  function handleReset() {
    setErrorMessage(null);
    setSuccessMessage(null);
    if (initialForm) {
      setForm(initialForm);
    } else {
      const defaults = defaultFormState();
      setForm(defaults);
      setInitialForm(defaults);
      setProfileId(null);
    }
    setErrors({});
  }

  return (
    <main className="min-h-screen bg-neutral-50 text-neutral-900 dark:bg-neutral-950 dark:text-neutral-50 flex flex-col">
      <header className="border-b border-neutral-200 px-4 py-3 flex items-center justify-between bg-white/80 text-neutral-800 dark:border-neutral-900 dark:bg-neutral-950/80 dark:text-neutral-100">
        <h1 className="text-sm font-medium">Settings</h1>
        <span className="text-xs text-neutral-500 dark:text-neutral-400">LLM and embeddings configuration</span>
      </header>
      <div className="max-w-3xl w-full mx-auto px-4 py-4 space-y-4 flex-1 flex flex-col">
        <section className="rounded-lg border border-neutral-200 bg-white/80 p-4 text-sm text-neutral-700 dark:border-neutral-900 dark:bg-neutral-950/60 dark:text-neutral-300">
          <h2 className="text-xs font-semibold text-neutral-500 dark:text-neutral-400 mb-2 uppercase tracking-wide">
            Model endpoints
          </h2>
          <p className="text-xs text-neutral-500 dark:text-neutral-400">
            Configure your local LLM and embedding model endpoints, including provider, model IDs, and base URLs.
          </p>
        </section>

        <section className="rounded-lg border border-neutral-200 bg-white/80 p-4 dark:border-neutral-900 dark:bg-neutral-950/60">
          {loading ? (
            <div className="text-sm text-neutral-500 dark:text-neutral-400">Loading settings...</div>
          ) : (
            <form onSubmit={handleSubmit} className="space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="space-y-1">
                  <label className="text-xs font-medium text-neutral-700 dark:text-neutral-300">
                    Profile name
                  </label>
                  <input
                    className="w-full rounded-md border border-neutral-200 bg-white px-3 py-2 text-sm outline-none focus:border-neutral-400 dark:border-neutral-800 dark:bg-neutral-900 dark:text-neutral-100"
                    value={form.name}
                    onChange={(e) => updateField("name", e.target.value)}
                    placeholder="local-default"
                  />
                  {errors.name && (
                    <p className="text-[11px] text-red-500">{errors.name}</p>
                  )}
                </div>
                <div className="flex items-center gap-2 mt-5 md:mt-7">
                  <input
                    id="is_default"
                    type="checkbox"
                    className="h-4 w-4 rounded border-neutral-300 text-neutral-900 focus:ring-neutral-500 dark:border-neutral-700 dark:bg-neutral-900"
                    checked={form.is_default}
                    onChange={(e) => updateField("is_default", e.target.checked)}
                  />
                  <label
                    htmlFor="is_default"
                    className="text-xs text-neutral-700 dark:text-neutral-300"
                  >
                    Make this the default configuration
                  </label>
                </div>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="space-y-1">
                  <label className="text-xs font-medium text-neutral-700 dark:text-neutral-300">
                    LLM provider
                  </label>
                  <input
                    className="w-full rounded-md border border-neutral-200 bg-white px-3 py-2 text-sm outline-none focus:border-neutral-400 dark:border-neutral-800 dark:bg-neutral-900 dark:text-neutral-100"
                    value={form.llm_provider}
                    onChange={(e) => updateField("llm_provider", e.target.value)}
                    placeholder="ollama"
                  />
                  {errors.llm_provider && (
                    <p className="text-[11px] text-red-500">{errors.llm_provider}</p>
                  )}
                </div>
                <div className="space-y-1">
                  <label className="text-xs font-medium text-neutral-700 dark:text-neutral-300">
                    LLM model
                  </label>
                  <input
                    className="w-full rounded-md border border-neutral-200 bg-white px-3 py-2 text-sm outline-none focus:border-neutral-400 dark:border-neutral-800 dark:bg-neutral-900 dark:text-neutral-100"
                    value={form.llm_model_id}
                    onChange={(e) => updateField("llm_model_id", e.target.value)}
                    placeholder="llama3"
                  />
                  {errors.llm_model_id && (
                    <p className="text-[11px] text-red-500">{errors.llm_model_id}</p>
                  )}
                </div>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="space-y-1">
                  <label className="text-xs font-medium text-neutral-700 dark:text-neutral-300">
                    LLM base URL
                  </label>
                  <input
                    className="w-full rounded-md border border-neutral-200 bg-white px-3 py-2 text-sm outline-none focus:border-neutral-400 dark:border-neutral-800 dark:bg-neutral-900 dark:text-neutral-100"
                    value={form.llm_base_url}
                    onChange={(e) => updateField("llm_base_url", e.target.value)}
                    placeholder="http://localhost:11434"
                  />
                  {errors.llm_base_url && (
                    <p className="text-[11px] text-red-500">{errors.llm_base_url}</p>
                  )}
                </div>
                <div className="space-y-1">
                  <label className="text-xs font-medium text-neutral-700 dark:text-neutral-300">
                    LLM API key (optional)
                  </label>
                  <input
                    type="password"
                    className="w-full rounded-md border border-neutral-200 bg-white px-3 py-2 text-sm outline-none focus:border-neutral-400 dark:border-neutral-800 dark:bg-neutral-900 dark:text-neutral-100"
                    value={form.llm_api_key}
                    onChange={(e) => updateField("llm_api_key", e.target.value)}
                    placeholder="Only if your LLM needs it"
                  />
                </div>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="space-y-1">
                  <label className="text-xs font-medium text-neutral-700 dark:text-neutral-300">
                    Embedding provider
                  </label>
                  <input
                    className="w-full rounded-md border border-neutral-200 bg-white px-3 py-2 text-sm outline-none focus:border-neutral-400 dark:border-neutral-800 dark:bg-neutral-900 dark:text-neutral-100"
                    value={form.embedding_provider}
                    onChange={(e) => updateField("embedding_provider", e.target.value)}
                    placeholder="local"
                  />
                  {errors.embedding_provider && (
                    <p className="text-[11px] text-red-500">{errors.embedding_provider}</p>
                  )}
                </div>
                <div className="space-y-1">
                  <label className="text-xs font-medium text-neutral-700 dark:text-neutral-300">
                    Embedding model
                  </label>
                  <input
                    className="w-full rounded-md border border-neutral-200 bg-white px-3 py-2 text-sm outline-none focus:border-neutral-400 dark:border-neutral-800 dark:bg-neutral-900 dark:text-neutral-100"
                    value={form.embedding_model_name}
                    onChange={(e) => updateField("embedding_model_name", e.target.value)}
                    placeholder="nomic-embed-text"
                  />
                  {errors.embedding_model_name && (
                    <p className="text-[11px] text-red-500">{errors.embedding_model_name}</p>
                  )}
                </div>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="space-y-1">
                  <label className="text-xs font-medium text-neutral-700 dark:text-neutral-300">
                    Embedding base URL
                  </label>
                  <input
                    className="w-full rounded-md border border-neutral-200 bg-white px-3 py-2 text-sm outline-none focus:border-neutral-400 dark:border-neutral-800 dark:bg-neutral-900 dark:text-neutral-100"
                    value={form.embedding_base_url}
                    onChange={(e) => updateField("embedding_base_url", e.target.value)}
                    placeholder="http://localhost:11434"
                  />
                  {errors.embedding_base_url && (
                    <p className="text-[11px] text-red-500">{errors.embedding_base_url}</p>
                  )}
                </div>
                <div className="space-y-1">
                  <label className="text-xs font-medium text-neutral-700 dark:text-neutral-300">
                    Embedding API key (optional)
                  </label>
                  <input
                    type="password"
                    className="w-full rounded-md border border-neutral-200 bg-white px-3 py-2 text-sm outline-none focus:border-neutral-400 dark:border-neutral-800 dark:bg-neutral-900 dark:text-neutral-100"
                    value={form.embedding_api_key}
                    onChange={(e) => updateField("embedding_api_key", e.target.value)}
                    placeholder="Only if your embedding service needs it"
                  />
                </div>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div className="space-y-1">
                  <label className="text-xs font-medium text-neutral-700 dark:text-neutral-300">
                    Retrieval top K
                  </label>
                  <input
                    className="w-full rounded-md border border-neutral-200 bg-white px-3 py-2 text-sm outline-none focus:border-neutral-400 dark:border-neutral-800 dark:bg-neutral-900 dark:text-neutral-100"
                    value={form.retrieval_top_k}
                    onChange={(e) => updateField("retrieval_top_k", e.target.value)}
                    placeholder="8"
                  />
                  {errors.retrieval_top_k && (
                    <p className="text-[11px] text-red-500">{errors.retrieval_top_k}</p>
                  )}
                </div>
                <div className="space-y-1">
                  <label className="text-xs font-medium text-neutral-700 dark:text-neutral-300">
                    Score threshold (optional)
                  </label>
                  <input
                    className="w-full rounded-md border border-neutral-200 bg-white px-3 py-2 text-sm outline-none focus:border-neutral-400 dark:border-neutral-800 dark:bg-neutral-900 dark:text-neutral-100"
                    value={form.retrieval_score_threshold}
                    onChange={(e) => updateField("retrieval_score_threshold", e.target.value)}
                    placeholder="e.g. 0.2"
                  />
                  {errors.retrieval_score_threshold && (
                    <p className="text-[11px] text-red-500">{errors.retrieval_score_threshold}</p>
                  )}
                </div>
                <div className="space-y-1">
                  <label className="text-xs font-medium text-neutral-700 dark:text-neutral-300">
                    Reranker model (optional)
                  </label>
                  <input
                    className="w-full rounded-md border border-neutral-200 bg-white px-3 py-2 text-sm outline-none focus:border-neutral-400 dark:border-neutral-800 dark:bg-neutral-900 dark:text-neutral-100"
                    value={form.reranker_model}
                    onChange={(e) => updateField("reranker_model", e.target.value)}
                    placeholder="Model name"
                  />
                </div>
              </div>

              <div className="flex items-center gap-3">
                <input
                  id="hybrid_search_enabled"
                  type="checkbox"
                  className="h-4 w-4 rounded border-neutral-300 text-neutral-900 focus:ring-neutral-500 dark:border-neutral-700 dark:bg-neutral-900"
                  checked={form.hybrid_search_enabled}
                  onChange={(e) => updateField("hybrid_search_enabled", e.target.checked)}
                />
                <label
                  htmlFor="hybrid_search_enabled"
                  className="text-xs text-neutral-700 dark:text-neutral-300"
                >
                  Enable hybrid search (sparse + dense)
                </label>
              </div>

              {(errorMessage || successMessage) && (
                <div className="text-xs">
                  {errorMessage && (
                    <p className="text-red-500">{errorMessage}</p>
                  )}
                  {successMessage && !errorMessage && (
                    <p className="text-emerald-500">{successMessage}</p>
                  )}
                </div>
              )}

              <div className="flex items-center gap-3">
                <button
                  type="submit"
                  disabled={saving}
                  className="text-xs font-medium px-4 py-2 rounded-full bg-neutral-900 text-neutral-50 disabled:opacity-40 dark:bg-neutral-100 dark:text-neutral-900"
                >
                  {saving ? "Saving" : "Save settings"}
                </button>
                <button
                  type="button"
                  onClick={handleReset}
                  disabled={saving}
                  className="text-xs font-medium px-4 py-2 rounded-full border border-neutral-300 text-neutral-700 hover:bg-neutral-100 disabled:opacity-40 dark:border-neutral-700 dark:text-neutral-200 dark:hover:bg-neutral-900"
                >
                  Reset
                </button>
              </div>
            </form>
          )}
        </section>
      </div>
    </main>
  );
}
