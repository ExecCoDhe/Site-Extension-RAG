const API_BASE_URL = "http://127.0.0.1:8000";
const STORAGE_KEY = "proj1Job";
const POLL_INTERVAL_MS = 1500;

const elements = {
  currentHost: document.getElementById("current-host"),
  indexedHost: document.getElementById("indexed-host"),
  pageCount: document.getElementById("page-count"),
  chunkCount: document.getElementById("chunk-count"),
  status: document.getElementById("status"),
  lastSync: document.getElementById("last-sync"),
  syncStats: document.getElementById("sync-stats"),
  message: document.getElementById("message"),
  hostWarning: document.getElementById("host-warning"),
  ackHostWarning: document.getElementById("ack-host-warning"),
  startButton: document.getElementById("start-button"),
  chatForm: document.getElementById("chat-form"),
  question: document.getElementById("question"),
  submitButton: document.getElementById("submit-button"),
  answer: document.getElementById("answer"),
  citations: document.getElementById("citations")
};

const state = {
  currentUrl: null,
  currentHost: null,
  workspace: null,
  run: null,
  mode: "idle",
  thinking: false,
  pollTimer: null,
  hostWarningAcknowledged: false
};

document.addEventListener("DOMContentLoaded", initialize);
elements.startButton.addEventListener("click", startIngest);
elements.chatForm.addEventListener("submit", submitQuestion);
elements.ackHostWarning.addEventListener("change", () => {
  state.hostWarningAcknowledged = elements.ackHostWarning.checked;
  render();
});
elements.question.addEventListener("input", render);

async function initialize() {
  const tab = await getActiveTab();
  state.currentUrl = tab?.url || null;
  state.currentHost = state.currentUrl ? getHostname(state.currentUrl) : null;

  render();
  await refreshWorkspace();
}

async function getActiveTab() {
  const tabs = await chrome.tabs.query({ active: true, currentWindow: true });
  return tabs[0];
}

async function startIngest() {
  if (!state.currentUrl) {
    setError("No active tab URL is available.");
    return;
  }

  setMessage("Starting ingest...");
  elements.startButton.disabled = true;

  const response = await apiFetch("/ingest", {
    method: "POST",
    body: JSON.stringify({ url: state.currentUrl })
  });

  if (response.error) {
    handleError(response.error);
    return;
  }

  state.run = response;
  state.mode = response.state;
  await saveState();
  render();
  pollStatus();
}

async function refreshWorkspace() {
  const response = await apiFetch("/workspace/status");
  if (response.error) {
    handleError(response.error);
    return;
  }
  state.workspace = response;
  state.mode = response.state;
  await saveState();
  render();
  if (response.active_run_id && response.state === "ingesting") {
    state.run = { run_id: response.active_run_id, job_id: response.active_run_id };
    pollStatus();
  }
}

async function refreshStatus() {
  const runId = state.run?.run_id || state.run?.job_id || state.workspace?.active_run_id;
  if (!runId) {
    await refreshWorkspace();
    return;
  }
  const response = await apiFetch(`/ingest/${runId}/status`);
  if (response.error) {
    handleError(response.error);
    return;
  }

  state.run = response;
  state.mode = response.state;
  await refreshWorkspace();
  await saveState();
  render();

  if (state.mode === "ingesting") {
    pollStatus();
  }
}

function pollStatus() {
  clearPollTimer();
  state.pollTimer = setTimeout(refreshStatus, POLL_INTERVAL_MS);
}

async function submitQuestion(event) {
  event.preventDefault();

  if (state.mode !== "ready") {
    handleError({
      code: "CHAT_BEFORE_READY",
      message: "Ingest a site before asking questions.",
      retryable: true
    });
    return;
  }

  state.thinking = true;
  setMessage("Thinking...");
  render();

  const response = await apiFetch("/chat", {
    method: "POST",
    body: JSON.stringify({
      workspace_id: state.workspace?.workspace_id,
      session_id: "default",
      question: elements.question.value.trim()
    })
  });

  state.thinking = false;

  if (response.error) {
    handleError(response.error);
    return;
  }

  renderAnswer(response);
  setMessage(messageForGroundedness(response.groundedness));
  render();
}

async function apiFetch(path, options = {}) {
  try {
    const response = await fetch(`${API_BASE_URL}${path}`, {
      headers: { "Content-Type": "application/json" },
      ...options
    });
    const body = await response.json();
    return body;
  } catch (_error) {
    return {
      error: {
        code: "BACKEND_UNAVAILABLE",
        message: "Backend is not running. Start the local server and try again.",
        details: null,
        retryable: true
      }
    };
  }
}

function handleError(error) {
  if (error.code === "ACTIVE_JOB" && error.details?.job_id) {
    state.run = error.details;
    state.mode = "ingesting";
    saveState();
    setMessage(error.message);
    render();
    pollStatus();
    return;
  }

  if (["INGEST_TIMEOUT", "NO_PAGES_INDEXED", "CHAT_BEFORE_READY"].includes(error.code)) {
    state.run = null;
    chrome.storage.local.remove(STORAGE_KEY);
  }

  state.mode = "error";
  setMessage(error.message);
  render();
}

async function saveState() {
  if (state.workspace) {
    await chrome.storage.local.set({ [STORAGE_KEY]: { workspace: state.workspace, run: state.run } });
  }
}

function render() {
  elements.currentHost.textContent = state.currentHost || "Unavailable";
  elements.indexedHost.textContent = state.workspace?.hostname || "None";
  elements.pageCount.textContent = String(state.workspace?.page_count || state.run?.page_count || 0);
  elements.chunkCount.textContent = String(state.workspace?.chunk_count || state.run?.chunk_count || 0);
  elements.status.textContent = state.thinking ? "Thinking" : titleCase(state.mode);
  elements.lastSync.textContent = state.workspace?.last_synced_at || "Never";
  elements.syncStats.textContent = syncStatsLabel(state.run);

  const ready = state.mode === "ready";
  const ingesting = state.mode === "ingesting";
  const hostMismatch = ready && state.currentHost && state.workspace?.hostname && state.currentHost !== state.workspace.hostname;
  const hostBlocked = hostMismatch && !state.hostWarningAcknowledged;

  elements.hostWarning.classList.toggle("hidden", !hostMismatch);
  elements.startButton.disabled = ingesting || state.thinking;
  elements.question.disabled = !ready || state.thinking;
  elements.submitButton.disabled = !ready || state.thinking || hostBlocked || !elements.question.value.trim();

  if (ingesting) {
    setMessage(`Syncing ${state.workspace?.hostname || state.run?.hostname || "site"} (${state.run?.page_count || 0} pages)...`);
  }
}

function renderAnswer(response) {
  elements.answer.textContent = response.answer || "No answer returned.";
  elements.citations.textContent = "";

  const seen = new Set();
  const citations = response.citations || response.evidence || [];
  if (citations.length === 0) {
    const item = document.createElement("li");
    item.textContent = "No citations returned.";
    elements.citations.appendChild(item);
    return;
  }

  for (const citation of citations) {
    const dedupeKey = citation.evidence_id || citation.chunk_id;
    if (seen.has(dedupeKey)) {
      continue;
    }
    seen.add(dedupeKey);

    const item = document.createElement("li");
    const link = document.createElement("a");
    const label = citation.title || citation.url;

    link.textContent = label;
    if (isSafeHttpUrl(citation.url)) {
      link.href = citation.url;
      link.target = "_blank";
      link.rel = "noreferrer";
    }

    const meta = document.createElement("div");
    meta.textContent = [
      citation.section,
      citation.chunk_id,
      `score ${Number(citation.rerank_score || citation.score || 0).toFixed(3)}`
    ].filter(Boolean).join(" · ");

    if (citation.snippet) {
      const snippet = document.createElement("blockquote");
      snippet.textContent = citation.snippet;
      item.appendChild(snippet);
    }

    if (citation.nearby_context && citation.nearby_context !== citation.snippet) {
      const details = document.createElement("details");
      const summary = document.createElement("summary");
      const nearby = document.createElement("blockquote");
      summary.textContent = "Nearby context";
      nearby.textContent = citation.nearby_context;
      details.appendChild(summary);
      details.appendChild(nearby);
      item.appendChild(details);
    }

    item.appendChild(link);
    item.appendChild(meta);
    elements.citations.appendChild(item);
  }
}

function messageForGroundedness(groundedness) {
  if (groundedness === "grounded") {
    return "Answer grounded in indexed content.";
  }
  if (groundedness === "partially_grounded") {
    return "Answer partially grounded; review the evidence.";
  }
  return "Not found in indexed content.";
}

function isSafeHttpUrl(url) {
  try {
    return ["http:", "https:"].includes(new URL(url).protocol);
  } catch (_error) {
    return false;
  }
}

function getHostname(url) {
  try {
    return new URL(url).hostname;
  } catch (_error) {
    return null;
  }
}

function setError(message) {
  state.mode = "error";
  setMessage(message);
  render();
}

function setMessage(message) {
  elements.message.textContent = message || "";
}

function syncStatsLabel(run) {
  if (!run) {
    return "None";
  }
  return [
    run.fetched_count != null ? `${run.fetched_count} fetched` : null,
    run.skipped_count != null ? `${run.skipped_count} skipped` : null,
    run.rendered_fallback_count != null ? `${run.rendered_fallback_count} rendered` : null
  ].filter(Boolean).join(" · ") || "None";
}

function clearPollTimer() {
  if (state.pollTimer) {
    clearTimeout(state.pollTimer);
    state.pollTimer = null;
  }
}

function titleCase(value) {
  return value.charAt(0).toUpperCase() + value.slice(1);
}
