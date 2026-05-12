const video = document.getElementById("video");
const overlay = document.getElementById("overlay");
const overlayCtx = overlay.getContext("2d");
const captureCanvas = document.getElementById("captureCanvas");
const captureCtx = captureCanvas.getContext("2d");
const guide = document.getElementById("guide");
const statusEl = document.getElementById("status");
const scoreValue = document.getElementById("scoreValue");
const meterFill = document.getElementById("meterFill");
const registryState = document.getElementById("registryState");
const backendState = document.getElementById("backendState");
const thresholdState = document.getElementById("thresholdState");
const patientList = document.getElementById("patientList");
const patientNameInput = document.getElementById("patientName");
const startButton = document.getElementById("startButton");
const auxButton = document.getElementById("auxButton");
const enrollButton = document.getElementById("enrollButton");
const detectButton = document.getElementById("detectButton");

const crop = [0.32, 0.08, 0.68, 0.96];
const DETECTION_HOLD_MS = 600;
const DETECTION_FADE_MS = 250;
let threshold = 0.74;
let stream = null;
let detecting = false;
let detectionTimer = null;
let lastDetection = null;
let lastDetectionAt = 0;
let backendName = "";

function setStatus(text, state = "idle") {
  statusEl.textContent = text;
  statusEl.dataset.state = state;
}

function setScore(score) {
  const safeScore = Number.isFinite(score) ? score : 0;
  scoreValue.textContent = safeScore.toFixed(2);
  meterFill.style.width = `${Math.max(0, Math.min(100, safeScore * 100))}%`;
}

async function api(path, body) {
  const response = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: body ? JSON.stringify(body) : "{}",
  });
  const payload = await response.json();
  if (!response.ok || payload.error) {
    throw new Error(payload.error || `Request failed: ${response.status}`);
  }
  return payload;
}

function renderPatients(patients) {
  patientList.innerHTML = "";
  for (const p of patients || []) {
    const li = document.createElement("li");
    const name = document.createElement("span");
    name.textContent = p.name;
    const count = document.createElement("span");
    count.className = "count";
    count.textContent = `${p.samples} samples`;
    li.appendChild(name);
    li.appendChild(count);
    patientList.appendChild(li);
  }
}

function configureAuxButton(payload) {
  const supportsLearned = /hog|haar/.test(payload.backend || "");
  if (supportsLearned) {
    auxButton.dataset.mode = "clear";
    auxButton.textContent = "Clear";
    auxButton.title = "Wipe registry + background";
  } else {
    auxButton.dataset.mode = "background";
    auxButton.textContent = payload.background_ready ? "Background ✓" : "Background";
    auxButton.title = "Capture empty room baseline";
  }
}

async function refreshStatus() {
  const response = await fetch("/api/status");
  const payload = await response.json();
  threshold = payload.threshold;
  backendName = payload.backend || "Unknown";
  if (payload.needs_reenrollment) {
    registryState.textContent = "Re-enroll";
  } else {
    registryState.textContent = payload.compatible_patient_count
      ? `${payload.compatible_patient_count} enrolled`
      : "Empty";
  }
  backendState.textContent = backendName;
  thresholdState.textContent = threshold.toFixed(2);
  renderPatients(payload.patients);
  configureAuxButton(payload);
}

async function startCamera() {
  stream = await navigator.mediaDevices.getUserMedia({
    video: { facingMode: "user", width: { ideal: 960 }, height: { ideal: 720 } },
    audio: false,
  });
  video.srcObject = stream;
  await video.play();
  auxButton.disabled = false;
  enrollButton.disabled = false;
  detectButton.disabled = false;
  setStatus("Camera ready", "idle");
  resizeOverlay();
  requestAnimationFrame(drawOverlay);
}

function resizeOverlay() {
  const bounds = video.getBoundingClientRect();
  overlay.width = Math.max(1, Math.round(bounds.width));
  overlay.height = Math.max(1, Math.round(bounds.height));
  guide.style.left = `${crop[0] * 100}%`;
  guide.style.top = `${crop[1] * 100}%`;
  guide.style.width = `${(crop[2] - crop[0]) * 100}%`;
  guide.style.height = `${(crop[3] - crop[1]) * 100}%`;
}

function bboxFreshness() {
  if (!lastDetection || !lastDetection.bbox) return 0;
  const age = performance.now() - lastDetectionAt;
  if (age < DETECTION_HOLD_MS) return 1;
  if (age < DETECTION_HOLD_MS + DETECTION_FADE_MS) {
    return 1 - (age - DETECTION_HOLD_MS) / DETECTION_FADE_MS;
  }
  return 0;
}

function drawOverlay() {
  resizeOverlay();
  overlayCtx.clearRect(0, 0, overlay.width, overlay.height);

  const alpha = bboxFreshness();
  if (alpha > 0 && lastDetection && lastDetection.bbox) {
    const [nx1, ny1, nx2, ny2] = lastDetection.bbox;
    const x = nx1 * overlay.width;
    const y = ny1 * overlay.height;
    const w = (nx2 - nx1) * overlay.width;
    const h = (ny2 - ny1) * overlay.height;
    const color = lastDetection.matched
      ? "#1a6f5a"
      : lastDetection.mode === "person" || lastDetection.mode === "face"
      ? "#c9a227"
      : "#a13030";

    overlayCtx.save();
    overlayCtx.globalAlpha = alpha;
    overlayCtx.lineWidth = 3;
    overlayCtx.setLineDash([10, 6]);
    overlayCtx.strokeStyle = color;
    overlayCtx.strokeRect(x, y, w, h);
    overlayCtx.setLineDash([]);

    const corner = Math.min(20, w * 0.18, h * 0.18);
    overlayCtx.lineWidth = 4;
    overlayCtx.beginPath();
    overlayCtx.moveTo(x, y + corner);
    overlayCtx.lineTo(x, y);
    overlayCtx.lineTo(x + corner, y);
    overlayCtx.moveTo(x + w - corner, y);
    overlayCtx.lineTo(x + w, y);
    overlayCtx.lineTo(x + w, y + corner);
    overlayCtx.moveTo(x + w, y + h - corner);
    overlayCtx.lineTo(x + w, y + h);
    overlayCtx.lineTo(x + w - corner, y + h);
    overlayCtx.moveTo(x + corner, y + h);
    overlayCtx.lineTo(x, y + h);
    overlayCtx.lineTo(x, y + h - corner);
    overlayCtx.stroke();

    const name = lastDetection.matched ? lastDetection.name : "no match";
    const label = `${name} · ${lastDetection.mode} ${lastDetection.score.toFixed(2)}`;
    overlayCtx.font = "bold 18px 'Patrick Hand', 'Comic Sans MS', sans-serif";
    const padX = 8;
    const textW = overlayCtx.measureText(label).width;
    const labelH = 24;
    const labelY = Math.max(0, y - labelH - 4);
    overlayCtx.fillStyle = color;
    overlayCtx.fillRect(x, labelY, textW + padX * 2, labelH);
    overlayCtx.fillStyle = "#fff";
    overlayCtx.fillText(label, x + padX, labelY + 18);
    overlayCtx.restore();
  }

  requestAnimationFrame(drawOverlay);
}

function captureFrame() {
  const targetWidth = 640;
  const scale = targetWidth / video.videoWidth;
  captureCanvas.width = targetWidth;
  captureCanvas.height = Math.round(video.videoHeight * scale);
  captureCtx.drawImage(video, 0, 0, captureCanvas.width, captureCanvas.height);
  return captureCanvas.toDataURL("image/jpeg", 0.82);
}

async function setBackground() {
  setStatus("Capturing background", "busy");
  const result = await api("/api/background", { image: captureFrame() });
  setScore(0);
  setStatus(result.background_ready ? "Background set" : "Background failed", result.background_ready ? "match" : "miss");
  await refreshStatus();
}

async function clearRegistry() {
  if (!confirm("Wipe enrolled patients and background?")) return;
  setStatus("Clearing", "busy");
  await api("/api/clear");
  setScore(0);
  lastDetection = null;
  setStatus("Cleared", "idle");
  await refreshStatus();
}

async function handleAux() {
  auxButton.disabled = true;
  enrollButton.disabled = true;
  detectButton.disabled = true;
  try {
    if (auxButton.dataset.mode === "clear") {
      await clearRegistry();
    } else {
      await setBackground();
    }
  } catch (error) {
    setStatus(error.message, "miss");
  }
  auxButton.disabled = false;
  enrollButton.disabled = false;
  detectButton.disabled = false;
}

async function enrollMe() {
  const rawName = (patientNameInput.value || "").trim();
  const name = rawName || "Me";
  enrollButton.disabled = true;
  detectButton.disabled = true;
  setStatus(`Enrolling ${name}`, "busy");
  const samples = [];
  for (let i = 0; i < 8; i += 1) {
    samples.push(captureFrame());
    await new Promise((resolve) => setTimeout(resolve, 180));
  }
  const result = await api("/api/enroll", { name, samples, crop });
  setStatus(`Enrolled ${result.name} (${result.reference_count})`, "match");
  patientNameInput.value = "";
  enrollButton.disabled = false;
  detectButton.disabled = false;
  await refreshStatus();
}

async function detectOnce() {
  if (!detecting) return;
  try {
    const result = await api("/api/detect", { image: captureFrame(), crop });
    setScore(result.confidence ?? result.score);
    if (result.needs_enrollment) {
      lastDetection = null;
      setStatus(result.needs_reenrollment ? "Re-enroll first" : "Enroll first", "idle");
    } else if (!result.person_present) {
      setStatus("No person in guide", "idle");
    } else {
      lastDetection = {
        bbox: result.identity_bbox,
        mode: result.identity_mode,
        score: result.score,
        matched: !!result.matched,
        name: result.patient_name || result.patient_id || "unknown",
      };
      lastDetectionAt = performance.now();
      if (result.matched) {
        setStatus(`${lastDetection.name} · ${result.identity_mode} ${result.score.toFixed(2)}`, "match");
      } else {
        setStatus(`Unknown · ${result.identity_mode} ${result.score.toFixed(2)}`, "miss");
      }
    }
  } catch (error) {
    setStatus(error.message, "miss");
  }
}

function toggleDetection() {
  detecting = !detecting;
  detectButton.textContent = detecting ? "Stop" : "Detect";
  if (detecting) {
    setStatus("Detecting", "busy");
    detectOnce();
    detectionTimer = setInterval(detectOnce, 500);
  } else {
    clearInterval(detectionTimer);
    detectionTimer = null;
    lastDetection = null;
    setStatus("Camera ready", "idle");
  }
}

startButton.addEventListener("click", async () => {
  try {
    startButton.disabled = true;
    await startCamera();
  } catch (error) {
    startButton.disabled = false;
    setStatus(error.message, "miss");
  }
});

enrollButton.addEventListener("click", () => {
  enrollMe().catch((error) => {
    enrollButton.disabled = false;
    detectButton.disabled = false;
    setStatus(error.message, "miss");
  });
});

auxButton.addEventListener("click", handleAux);

detectButton.addEventListener("click", toggleDetection);
window.addEventListener("resize", resizeOverlay);
refreshStatus().catch(() => {
  registryState.textContent = "Unavailable";
  backendState.textContent = "Unavailable";
});
