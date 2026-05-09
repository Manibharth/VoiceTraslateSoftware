/* ── VoiceTranslate AI ─────────────────────────────────────────────────────── */

const API = "";

const STATE = {
  jobId:    null,
  isVideo:  false,
  pollTimer: null,
};

const SPEAKER_COLORS = [
  "#4f6ef7","#ec4899","#22c55e","#f59e0b",
  "#a855f7","#06b6d4","#f97316","#14b8a6",
];

const VOICE_PROFILES = [
  { id: "Hero Male",      label: "Hero Male",      desc: "Deep & powerful" },
  { id: "Heroine Female", label: "Heroine Female", desc: "Soft & clear" },
  { id: "Child Voice",    label: "Child Voice",    desc: "High-pitched" },
  { id: "Comedian Voice", label: "Comedian Voice", desc: "Energetic" },
];

const STEPS = ["diarizing","analyzing","transcribing","translating","synthesizing","merging","done"];

// ── Init ───────────────────────────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
  initDrop();
  initLangButtons();
  document.getElementById("file-input").addEventListener("change", e => {
    if (e.target.files[0]) handleFile(e.target.files[0]);
  });
});

function initLangButtons() {
  document.querySelectorAll("#src-lang-options .lang-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      document.querySelectorAll("#src-lang-options .lang-btn").forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
    });
  });
  document.querySelectorAll("#tgt-lang-options .lang-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      document.querySelectorAll("#tgt-lang-options .lang-btn").forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      // Update subtitle toggle label
      const name = btn.textContent.trim().split(" ").slice(1).join(" ");
      const lbl = document.querySelector(".toggle-label");
      if (lbl) lbl.textContent = `Burn ${name} subtitles into video`;
    });
  });
}

function getSelectedLang(groupId) {
  const active = document.querySelector(`#${groupId} .lang-btn.active`);
  return active ? active.dataset.code : (groupId === "tgt-lang-options" ? "ta" : "auto");
}

// ── Drag & Drop ────────────────────────────────────────────────────────────────
function initDrop() {
  const zone = document.getElementById("drop-area");
  zone.addEventListener("click", () => document.getElementById("file-input").click());
  zone.addEventListener("dragover", e => { e.preventDefault(); zone.classList.add("drag-over"); });
  zone.addEventListener("dragleave", () => zone.classList.remove("drag-over"));
  zone.addEventListener("drop", e => {
    e.preventDefault();
    zone.classList.remove("drag-over");
    if (e.dataTransfer.files[0]) handleFile(e.dataTransfer.files[0]);
  });
}

function handleFile(file) {
  const ext = file.name.split(".").pop().toLowerCase();
  const allowed = ["mp4","mkv","avi","mov","mp3","wav","m4a","ogg","webm"];
  if (!allowed.includes(ext)) { toast("Unsupported file type", "error"); return; }

  window._file = file;
  STATE.isVideo = ["mp4","mkv","avi","mov","webm"].includes(ext);

  // Show file info
  document.getElementById("drop-area").classList.add("hidden");
  const sel = document.getElementById("file-selected");
  sel.classList.remove("hidden");
  document.getElementById("file-name").textContent = file.name;
  document.getElementById("file-size").textContent = formatBytes(file.size);
  document.getElementById("file-thumb").textContent = STATE.isVideo ? "🎬" : "🎵";

  // Media preview
  const url = URL.createObjectURL(file);
  const prev = document.getElementById("media-preview");
  prev.classList.remove("hidden");
  if (STATE.isVideo) {
    const v = document.getElementById("preview-video");
    v.src = url; v.classList.remove("hidden");
    document.getElementById("preview-audio").classList.add("hidden");
  } else {
    const a = document.getElementById("preview-audio");
    a.src = url; a.classList.remove("hidden");
    document.getElementById("preview-video").classList.add("hidden");
  }

  document.getElementById("btn-start").disabled = false;
}

function resetAll() {
  window._file = null;
  STATE.jobId = null;
  if (STATE.pollTimer) clearInterval(STATE.pollTimer);

  document.getElementById("drop-area").classList.remove("hidden");
  document.getElementById("file-selected").classList.add("hidden");
  document.getElementById("file-input").value = "";
  document.getElementById("btn-start").disabled = true;

  ["panel-progress","panel-speakers","panel-output"].forEach(id =>
    document.getElementById(id).classList.add("hidden")
  );
}

// ── Pipeline ───────────────────────────────────────────────────────────────────
async function startPipeline() {
  if (!window._file) return;

  const btn = document.getElementById("btn-start");
  btn.disabled = true;
  btn.innerHTML = `<span class="spin"></span> Uploading…`;

  try {
    const form = new FormData();
    form.append("file", window._file);
    const res  = await fetch(`${API}/api/upload`, { method: "POST", body: form });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "Upload failed");

    STATE.jobId = data.job_id;
    show("panel-progress");

    const targetLang = getSelectedLang("tgt-lang-options");
    const sourceLang = getSelectedLang("src-lang-options");
    STATE.targetLang = targetLang;

    await fetch(`${API}/api/process_full`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        job_id:            STATE.jobId,
        model_size:        document.getElementById("model-size").value,
        hf_token:          document.getElementById("hf-token").value.trim() || null,
        include_subtitles: document.getElementById("include-subtitles").checked,
        target_lang:       targetLang,
        source_lang:       sourceLang,
      }),
    });

    STATE.pollTimer = setInterval(() => pollStatus(STATE.jobId), 2500);

  } catch (err) {
    toast(err.message, "error");
    btn.disabled = false;
    btn.innerHTML = `<svg width="20" height="20" viewBox="0 0 24 24" fill="none"><polygon points="5,3 19,12 5,21" fill="currentColor"/></svg> Start AI Translation`;
  }
}

// ── Polling ────────────────────────────────────────────────────────────────────
async function pollStatus(jobId) {
  try {
    const data = await fetch(`${API}/api/status/${jobId}`).then(r => r.json());
    updatePipeline(data);

    if (data.step === "done") {
      clearInterval(STATE.pollTimer);
      await loadDashboard(jobId);
      show("panel-speakers");
      await buildOutput(jobId);
      show("panel-output");
      toast("Translation complete!", "success");
    } else if (data.step === "failed") {
      clearInterval(STATE.pollTimer);
      toast("Pipeline failed: " + (data.message || "Unknown error"), "error");
      document.getElementById("btn-start").disabled = false;
      document.getElementById("btn-start").innerHTML = `<svg width="20" height="20" viewBox="0 0 24 24" fill="none"><polygon points="5,3 19,12 5,21" fill="currentColor"/></svg> Start AI Translation`;
    }
  } catch (e) {
    console.error("Poll error", e);
  }
}

function updatePipeline(data) {
  const step    = data.step || "";
  const pct     = data.progress || 0;
  const msg     = data.message || "";
  const curIdx  = STEPS.indexOf(step);

  document.getElementById("progress-fill").style.width = pct + "%";
  document.getElementById("progress-pct").textContent  = pct + "%";
  document.getElementById("progress-headline").textContent = msg;

  document.querySelectorAll(".pipe-step").forEach(el => {
    const s   = el.dataset.step;
    const idx = STEPS.indexOf(s);
    const st  = el.querySelector(".pipe-status");
    el.classList.remove("active","done","failed");

    if (step === "failed") {
      if (idx < curIdx)      { el.classList.add("done");   st.textContent = "✓ Done"; }
      else if (idx === curIdx){ el.classList.add("failed"); st.textContent = "✗ Failed"; }
      else                   { st.textContent = "—"; }
    } else if (step === "done" || idx < curIdx) {
      el.classList.add("done");
      st.textContent = "✓ Done";
    } else if (idx === curIdx) {
      el.classList.add("active");
      st.innerHTML = `<span class="spin"></span>Running`;
    } else {
      st.textContent = "—";
    }
  });
}

// ── Speaker Dashboard ──────────────────────────────────────────────────────────
async function loadDashboard(jobId) {
  const [spkRes, segRes] = await Promise.all([
    fetch(`${API}/api/speakers/${jobId}`).then(r => r.json()),
    fetch(`${API}/api/segments/${jobId}`).then(r => r.json()),
  ]);
  renderSpeakers(spkRes.speakers || [], jobId);
  renderTranscript(segRes.segments || []);
}

function renderSpeakers(speakers, jobId) {
  const grid = document.getElementById("speaker-cards");
  grid.innerHTML = "";

  speakers.forEach((spk, i) => {
    const color  = SPEAKER_COLORS[i % SPEAKER_COLORS.length];
    const avatar = spk.age_type === "Child" ? "🧒" : (spk.gender === "Female" ? "👩" : "🧔");

    const card = document.createElement("div");
    card.className = "spk-card";
    card.style.setProperty("--spk-color", color);
    card.querySelector ? null : null;

    card.innerHTML = `
      <style>#spk-card-${i}::before { background: ${color}; }</style>
      <span class="spk-avatar">${avatar}</span>
      <div class="spk-name" style="color:${color}">${spk.label}</div>
      <div class="spk-tags">
        <span class="spk-tag ${spk.gender === 'Female' ? 'tag-female' : 'tag-male'}">${spk.gender}</span>
        <span class="spk-tag ${spk.age_type === 'Child' ? 'tag-child' : 'tag-adult'}">${spk.age_type}</span>
        <span class="spk-tag tag-tone">${spk.tone || 'Neutral'}</span>
      </div>
      <div class="spk-pitch">Pitch: ${spk.pitch_mean ? spk.pitch_mean.toFixed(0) + ' Hz' : '—'}</div>
      <div class="spk-field-label">Voice Profile</div>
      <select class="spk-select" onchange="overrideVoice('${jobId}','${spk.label}',this.value)">
        ${VOICE_PROFILES.map(p => `<option value="${p.id}" ${p.id === spk.voice_profile ? 'selected' : ''}>${p.label} — ${p.desc}</option>`).join('')}
      </select>
      <button class="btn-preview" onclick="previewVoice('${jobId}','${spk.label}')">
        <svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor"><polygon points="5,3 19,12 5,21"/></svg>
        Preview Voice
      </button>
    `;
    card.id = `spk-card-${i}`;
    // Apply top bar color via inline style
    card.style.cssText += `--bar:${color}`;
    // Top color bar
    const bar = document.createElement("div");
    bar.style.cssText = `position:absolute;top:0;left:0;right:0;height:3px;background:${color};border-radius:10px 10px 0 0`;
    card.prepend(bar);

    grid.appendChild(card);
  });
}

function renderTranscript(segments) {
  const body = document.getElementById("transcript-body");
  body.innerHTML = "";
  const colorMap = {};
  let ci = 0;

  segments.forEach(seg => {
    if (!colorMap[seg.speaker]) colorMap[seg.speaker] = SPEAKER_COLORS[ci++ % SPEAKER_COLORS.length];
    const color = colorMap[seg.speaker];

    const row = document.createElement("div");
    row.className = "tr-row";
    row.innerHTML = `
      <div><span class="tr-spk" style="background:${color}22;color:${color};border:1px solid ${color}44">${seg.speaker}</span></div>
      <div class="tr-time">${fmtTime(seg.start)}</div>
      <div class="tr-en">${esc(seg.original || "")}</div>
      <div class="tr-ta">${esc(seg.translated || "")}</div>
    `;
    body.appendChild(row);
  });
}

async function overrideVoice(jobId, label, profile) {
  try {
    await fetch(`${API}/api/update_voice`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ job_id: jobId, speaker_label: label, voice_profile: profile }),
    });
    toast(`${label} → ${profile}`, "success");
  } catch {
    toast("Failed to update voice", "error");
  }
}

async function previewVoice(jobId, label) {
  try {
    const res = await fetch(`${API}/api/preview_voice/${jobId}/${label}`);
    if (!res.ok) { toast("Preview not available yet", "warning"); return; }
    const blob = await res.blob();
    new Audio(URL.createObjectURL(blob)).play();
    toast(`Playing ${label} voice…`);
  } catch {
    toast("Preview unavailable", "warning");
  }
}

// ── Output ─────────────────────────────────────────────────────────────────────
const LANG_NAMES = {
  "ta": "Tamil 🇮🇳", "en": "English 🇬🇧", "ja": "Japanese 🇯🇵",
  "zh-CN": "Chinese 🇨🇳", "th": "Thai 🇹🇭",
};

async function buildOutput(jobId) {
  const dlUrl   = `${API}/api/download/${jobId}`;
  const srtUrl  = `${API}/api/download_subtitle/${jobId}`;
  const langLabel = LANG_NAMES[STATE.targetLang] || STATE.targetLang || "Tamil 🇮🇳";

  const player = document.getElementById("output-player-wrap");
  if (STATE.isVideo) {
    player.innerHTML = `<video controls><source src="${dlUrl}" type="video/mp4">Your browser does not support video.</video>`;
  } else {
    player.innerHTML = `<audio controls style="width:100%"><source src="${dlUrl}">Your browser does not support audio.</audio>`;
  }

  const dl = document.getElementById("download-list");
  dl.innerHTML = `
    <a class="btn-download" href="${dlUrl}" download>
      <span class="dl-icon">${STATE.isVideo ? "🎬" : "🎵"}</span>
      <span class="dl-info">
        <div class="dl-label">Dubbed ${STATE.isVideo ? "Video" : "Audio"}</div>
        <div class="dl-sub">${langLabel} · AAC 192kbps</div>
      </span>
      <span class="dl-arrow">↓</span>
    </a>
    <a class="btn-download" href="${srtUrl}" download>
      <span class="dl-icon">📄</span>
      <span class="dl-info">
        <div class="dl-label">${langLabel} Subtitles</div>
        <div class="dl-sub">SRT format · UTF-8 encoded</div>
      </span>
      <span class="dl-arrow">↓</span>
    </a>
  `;

  // Stats
  try {
    const segRes = await fetch(`${API}/api/segments/${jobId}`).then(r => r.json());
    const spkRes = await fetch(`${API}/api/speakers/${jobId}`).then(r => r.json());
    const segs   = segRes.segments || [];
    const spks   = spkRes.speakers || [];
    const dur    = segs.length ? segs[segs.length-1].end : 0;
    document.getElementById("output-stats").innerHTML = `
      <div class="stat-row"><span>Speakers detected</span><span class="stat-val">${spks.length}</span></div>
      <div class="stat-row"><span>Segments translated</span><span class="stat-val">${segs.length}</span></div>
      <div class="stat-row"><span>Duration</span><span class="stat-val">${fmtTime(dur)}</span></div>
      <div class="stat-row"><span>Output language</span><span class="stat-val">Tamil 🇮🇳</span></div>
    `;
  } catch {}
}

// ── Helpers ────────────────────────────────────────────────────────────────────
function show(id) {
  const el = document.getElementById(id);
  el.classList.remove("hidden");
  setTimeout(() => el.scrollIntoView({ behavior: "smooth", block: "start" }), 50);
}

function fmtTime(sec) {
  if (!sec && sec !== 0) return "—";
  const m = Math.floor(sec / 60).toString().padStart(2, "0");
  const s = Math.floor(sec % 60).toString().padStart(2, "0");
  return `${m}:${s}`;
}

function formatBytes(b) {
  if (b < 1024)           return b + " B";
  if (b < 1024 * 1024)    return (b / 1024).toFixed(1) + " KB";
  return (b / 1024 / 1024).toFixed(1) + " MB";
}

function esc(s) {
  return String(s).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");
}

function toast(msg, type = "info") {
  const stack = document.getElementById("toast-stack");
  const el    = document.createElement("div");
  el.className = `toast ${type}`;
  el.innerHTML = `<span class="toast-dot"></span><span class="toast-msg">${esc(msg)}</span>`;
  stack.appendChild(el);
  setTimeout(() => el.remove(), 4500);
}
