let conversationHistory = [];

async function sendMessage() {
  const textarea = document.getElementById("message-input");
  const text = textarea.value.trim();
  if (!text) return;

  appendMessage("user", text);
  textarea.value = "";

  const payload = {
    message: text,
    history: conversationHistory,
    conversation_id: null,
  };

  try {
    const resp = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (!resp.ok) {
      appendMessage("assistant", "Error: " + resp.statusText);
      return;
    }

    const data = await resp.json();
    appendMessage("assistant", data.answer, data.sources);
  } catch (err) {
    appendMessage("assistant", "Error: " + err);
  }
}

function appendMessage(role, content, sources) {
  const messages = document.getElementById("messages");
  const msgEl = document.createElement("div");
  msgEl.className = "message " + role;

  const body = document.createElement("div");
  body.className = "message-body";
  body.textContent = content;
  msgEl.appendChild(body);

  if (sources && sources.length) {
    const srcList = document.createElement("ul");
    srcList.className = "sources";
    for (const s of sources) {
      const li = document.createElement("li");
      li.textContent = `${s.file_name || s.doc_id || "source"} (#${s.chunk_index ?? "?"})`;
      srcList.appendChild(li);
    }
    msgEl.appendChild(srcList);
  }

  messages.appendChild(msgEl);
  messages.scrollTop = messages.scrollHeight;

  conversationHistory.push({ role, content });
}

async function uploadDocument() {
  const input = document.getElementById("file-input");
  if (!input.files.length) return;
  const file = input.files[0];

  const formData = new FormData();
  formData.append("file", file);

  try {
    const resp = await fetch("/api/documents", {
      method: "POST",
      body: formData,
    });
    if (!resp.ok) {
      appendMessage("assistant", "Upload failed: " + resp.statusText);
      return;
    }
    appendMessage("assistant", "Document uploaded and indexed.");
  } catch (err) {
    appendMessage("assistant", "Upload error: " + err);
  }
}

async function loadConfig() {
  try {
    const resp = await fetch("/api/config");
    if (!resp.ok) return;
    const cfg = await resp.json();
    const form = document.getElementById("config-form");
    const mapping = {
      "llm.provider": cfg.llm?.provider,
      "llm.base_url": cfg.llm?.base_url,
      "llm.model": cfg.llm?.model,
      "embeddings.provider": cfg.embeddings?.provider,
      "embeddings.base_url": cfg.embeddings?.base_url,
      "embeddings.model": cfg.embeddings?.model,
      "storage.vector_dir": cfg.storage?.vector_dir,
      "storage.docs_dir": cfg.storage?.docs_dir,
      "rag.top_k": cfg.rag?.top_k,
      "rag.chunk_size": cfg.rag?.chunk_size,
      "rag.chunk_overlap": cfg.rag?.chunk_overlap,
    };
    for (const [name, value] of Object.entries(mapping)) {
      const input = form.querySelector(`input[name="${name}"]`);
      if (input && value != null) {
        input.value = value;
      }
    }
  } catch (err) {
    console.error("Failed to load config", err);
  }
}

async function saveConfig(event) {
  event.preventDefault();
  const form = event.target;
  const formData = new FormData(form);
  const update = {
    llm: {},
    embeddings: {},
    storage: {},
    rag: {},
  };

  for (const [key, value] of formData.entries()) {
    if (!value) continue;
    const [section, field] = key.split(".");
    if (!update[section]) continue;
    update[section][field] = isFinite(value) && value !== "" ? Number(value) : value;
  }

  try {
    const resp = await fetch("/api/config", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(update),
    });
    if (!resp.ok) {
      appendMessage("assistant", "Config save failed: " + resp.statusText);
      return;
    }
    appendMessage("assistant", "Config saved.");
  } catch (err) {
    appendMessage("assistant", "Config error: " + err);
  }
}

function setupUI() {
  document.getElementById("send-btn").addEventListener("click", sendMessage);
  document
    .getElementById("message-input")
    .addEventListener("keydown", (e) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
      }
    });
  document.getElementById("upload-btn").addEventListener("click", uploadDocument);
  document.getElementById("config-form").addEventListener("submit", saveConfig);
  document.getElementById("new-chat-btn").addEventListener("click", () => {
    conversationHistory = [];
    document.getElementById("messages").innerHTML = "";
  });

  loadConfig();
}

window.addEventListener("DOMContentLoaded", setupUI);
