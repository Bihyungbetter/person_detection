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
const backgroundState = document.getElementById("backgroundState");
const backendState = document.getElementById("backendState");
const thresholdState = document.getElementById("thresholdState");
const startButton = document.getElementById("startButton");
const backgroundButton = document.getElementById("backgroundButton");
const enrollButton = document.getElementById("enrollButton");
const detectButton = document.getElementById("detectButton");

const crop = [0.32, 0.08, 0.68, 0.96];
let threshold = 0.74;
let stream = null;
let detecting = false;
let detectionTimer = null;

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
    body: JSON.stringify(body),
  });
  const payload = await response.json();
  if (!response.ok || payload.error) {
    throw new Error(payload.error || `Request failed: ${response.status}`);
  }
  return payload;
}

async function refreshStatus() {
  const response = await fetch("/api/status");
  const payload = await response.json();
  threshold = payload.threshold;
  if (payload.needs_reenrollment) {
    registryState.textContent = "Re-enroll";
  } else {
    registryState.textContent = payload.compatible_patient_count ? "Enrolled" : "Empty";
  }
  backgroundState.textContent = payload.background_ready ? "Set" : "Not set";
  backendState.textContent = payload.backend || "Unknown";
  thresholdState.textContent = threshold.toFixed(2);
}

async function startCamera() {
  stream = await navigator.mediaDevices.getUserMedia({
    video: { facingMode: "user", width: { ideal: 960 }, height: { ideal: 720 } },
    audio: false,
  });
  video.srcObject = stream;
  await video.play();
  backgroundButton.disabled = false;
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

function drawOverlay() {
  resizeOverlay();
  overlayCtx.clearRect(0, 0, overlay.width, overlay.height);
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
  backgroundButton.disabled = true;
  enrollButton.disabled = true;
  detectButton.disabled = true;
  setStatus("Capturing background", "busy");
  const result = await api("/api/background", { image: captureFrame() });
  backgroundState.textContent = result.background_ready ? "Set" : "Not set";
  setScore(0);
  setStatus("Background set", "match");
  backgroundButton.disabled = false;
  enrollButton.disabled = false;
  detectButton.disabled = false;
}

async function enrollMe() {
  enrollButton.disabled = true;
  detectButton.disabled = true;
  setStatus("Enrolling", "busy");
  const samples = [];
  for (let i = 0; i < 8; i += 1) {
    samples.push(captureFrame());
    await new Promise((resolve) => setTimeout(resolve, 180));
  }
  const result = await api("/api/enroll", {
    patient_id: "me",
    name: "Me",
    samples,
    crop,
  });
  registryState.textContent = `${result.reference_count} samples`;
  setStatus("Enrolled", "match");
  enrollButton.disabled = false;
  detectButton.disabled = false;
}

async function detectOnce() {
  if (!detecting) return;
  try {
    const result = await api("/api/detect", { image: captureFrame(), crop });
    setScore(result.confidence ?? result.score);
    if (result.needs_enrollment) {
      setStatus(result.needs_reenrollment ? "Re-enroll first" : "Enroll first", "idle");
    } else if (!result.person_present) {
      setStatus("No person in guide", "idle");
    } else if (result.matched) {
      setStatus(`Detected ${result.patient_name || result.patient_id} · ${result.identity_mode} · ${result.score.toFixed(2)}`, "match");
    } else {
      setStatus(`Not detected · ${result.identity_mode} · ${result.score.toFixed(2)}`, "miss");
    }
  } catch (error) {
    setStatus(error.message, "miss");
  }
}

function toggleDetection() {
  detecting = !detecting;
  detectButton.textContent = detecting ? "Stop Detecting" : "Start Detecting";
  if (detecting) {
    setStatus("Detecting", "busy");
    detectOnce();
    detectionTimer = setInterval(detectOnce, 650);
  } else {
    clearInterval(detectionTimer);
    detectionTimer = null;
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

backgroundButton.addEventListener("click", () => {
  setBackground().catch((error) => {
    backgroundButton.disabled = false;
    enrollButton.disabled = false;
    detectButton.disabled = false;
    setStatus(error.message, "miss");
  });
});

detectButton.addEventListener("click", toggleDetection);
window.addEventListener("resize", resizeOverlay);
refreshStatus().catch(() => {
  registryState.textContent = "Unavailable";
  backgroundState.textContent = "Unavailable";
  backendState.textContent = "Unavailable";
});
