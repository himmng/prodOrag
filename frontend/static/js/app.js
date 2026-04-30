async function sendMessage() {
  const promptEl = document.getElementById("prompt");
  const messagesEl = document.getElementById("messages");
  const text = promptEl.value.trim();
  if (!text) return;

  const userBubble = document.createElement("div");
  userBubble.className = "message user";
  userBubble.textContent = text;
  messagesEl.appendChild(userBubble);
  messagesEl.scrollTop = messagesEl.scrollHeight;

  promptEl.value = "";

  const assistantBubble = document.createElement("div");
  assistantBubble.className = "message assistant";
  messagesEl.appendChild(assistantBubble);

  const payload = { messages: [{ role: "user", content: text }] };

  const resp = await fetch("/api/chat/stream", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  const reader = resp.body.getReader();
  const decoder = new TextDecoder();

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    assistantBubble.textContent += decoder.decode(value, { stream: true });
    messagesEl.scrollTop = messagesEl.scrollHeight;
  }
}

function setup() {
  document.getElementById("send").addEventListener("click", sendMessage);
  document
    .getElementById("prompt")
    .addEventListener("keydown", (e) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
      }
    });

  const fileInput = document.getElementById("file-input");
  fileInput.addEventListener("change", async () => {
    const files = fileInput.files;
    if (!files.length) return;
    const form = new FormData();
    for (const f of files) form.append("files", f);
    await fetch("/api/documents/upload", { method: "POST", body: form });
    fileInput.value = "";
  });
}

window.addEventListener("DOMContentLoaded", setup);
