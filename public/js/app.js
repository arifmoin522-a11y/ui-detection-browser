/**
 * Application entry point
 * Supports two modes: Upload Screenshot and Live Screen Detection
 */

// --- DOM refs ---
const imageInput = document.getElementById("imageInput");
const detectBtn = document.getElementById("detectBtn");
const statusEl = document.getElementById("status");
const canvas = document.getElementById("outputCanvas");
const loadingOverlay = document.getElementById("loadingOverlay");
const resultsEl = document.getElementById("results");
const resultsList = document.getElementById("resultsList");
const confSlider = document.getElementById("confThreshold");
const confValue = document.getElementById("confValue");

// Mode switcher
const modeUploadBtn = document.getElementById("modeUpload");
const modeLiveBtn = document.getElementById("modeLive");
const uploadPanel = document.getElementById("uploadPanel");
const livePanel = document.getElementById("livePanel");
const uploadCanvasWrap = document.getElementById("uploadCanvasWrap");

// Live mode elements
const startLiveBtn = document.getElementById("startLiveBtn");
const stopLiveBtn = document.getElementById("stopLiveBtn");
const liveConfSlider = document.getElementById("liveConfThreshold");
const liveConfValue = document.getElementById("liveConfValue");
const fpsLimitSlider = document.getElementById("fpsLimit");
const fpsLimitValue = document.getElementById("fpsLimitValue");
const liveContainer = document.getElementById("liveContainer");
const liveVideo = document.getElementById("liveVideo");
const liveOverlay = document.getElementById("liveOverlay");
const liveIndicator = document.getElementById("liveIndicator");
const fpsCounter = document.getElementById("fpsCounter");
const liveResults = document.getElementById("liveResults");
const liveResultsList = document.getElementById("liveResultsList");
const liveCount = document.getElementById("liveCount");

// --- State ---
let currentImage = null;
let detectionHistory = [];
const detector = new UIDetector();

let liveStream = null;
let liveAnimFrame = null;
let liveRunning = false;
let lastFrameTime = 0;
let frameCount = 0;
let fpsUpdateTime = 0;
let currentFps = 0;
let liveDetections = [];

// --- Mode Switching ---
modeUploadBtn.addEventListener("click", () => switchMode("upload"));
modeLiveBtn.addEventListener("click", () => switchMode("live"));

function switchMode(mode) {
    if (mode === "live" && liveRunning) return;
    if (mode === "upload" && liveRunning) stopLiveDetection();

    modeUploadBtn.classList.toggle("active", mode === "upload");
    modeLiveBtn.classList.toggle("active", mode === "live");
    modeUploadBtn.setAttribute("aria-selected", mode === "upload");
    modeLiveBtn.setAttribute("aria-selected", mode === "live");

    uploadPanel.classList.toggle("hidden", mode !== "upload");
    livePanel.classList.toggle("hidden", mode !== "live");
    uploadCanvasWrap.classList.toggle("hidden", mode !== "upload");
    liveContainer.classList.toggle("hidden", mode !== "live");
    resultsEl.classList.add("hidden");

    if (mode === "upload") {
        setStatus("Select an image to begin");
    } else {
        setStatus("Click 'Start Live Detection' to capture your screen");
    }
}

// --- Upload Mode ---
function resetView() {
    YoloVisionAdapter.resetCounter();
    detectionHistory = [];
    currentImage = null;
    canvas.width = 0;
    canvas.height = 0;
    const ctx = canvas.getContext("2d");
    if (ctx) ctx.clearRect(0, 0, 0, 0);
    resultsEl.classList.add("hidden");
    resultsList.innerHTML = "";
    detectBtn.disabled = true;
    imageInput.value = "";
    setStatus("View reset. Upload a new screenshot.", "");
}

function setStatus(msg, type = "") {
    statusEl.textContent = msg;
    statusEl.className = "status " + type;
}

async function initModel() {
    try {
        loadingOverlay.classList.remove("hidden");
        setStatus("Loading YOLOv8s model...");
        await detector.loadModel();
        setStatus("Model loaded. Select a screenshot to detect UI elements.", "success");
        loadingOverlay.classList.add("hidden");
    } catch (err) {
        setStatus("Failed to load model: " + err.message, "error");
        loadingOverlay.classList.add("hidden");
        console.error(err);
    }
}

imageInput.addEventListener("change", (e) => {
    const file = e.target.files[0];
    if (!file) return;

    const img = new Image();
    img.onload = () => {
        currentImage = img;
        detectBtn.disabled = false;

        canvas.width = img.width;
        canvas.height = img.height;
        const ctx = canvas.getContext("2d");
        ctx.drawImage(img, 0, 0);

        resultsEl.classList.add("hidden");
        resultsList.innerHTML = "";
        setStatus(`Image loaded (${img.width}x${img.height}). Click Detect.`, "success");
    };
    img.src = URL.createObjectURL(file);
});

detectBtn.addEventListener("click", async () => {
    if (!currentImage || !detector.session) return;

    try {
        detectBtn.disabled = true;
        setStatus("Detecting UI elements...");
        loadingOverlay.classList.remove("hidden");

        detector.confThreshold = parseFloat(confSlider.value);

        const detections = await detector.detect(currentImage);
        detector.drawDetections(canvas, currentImage, detections);

        const visionOutput = YoloVisionAdapter.adaptDetections(detections);

        detectionHistory.push({
            timestamp: Date.now(),
            inputSize: { width: currentImage.width, height: currentImage.height },
            count: visionOutput.length,
            elements: visionOutput.map((d) => d.type)
        });

        renderResults(visionOutput);
        setStatus(`Found ${visionOutput.length} UI element(s)`, "success");
    } catch (err) {
        setStatus("Detection failed: " + err.message, "error");
        console.error(err);
    } finally {
        detectBtn.disabled = false;
        loadingOverlay.classList.add("hidden");
    }
});

confSlider.addEventListener("input", () => {
    confValue.textContent = confSlider.value;
});

function renderResults(detections) {
    if (detections.length === 0) {
        resultsEl.classList.add("hidden");
        return;
    }

    resultsEl.classList.remove("hidden");
    resultsList.innerHTML = "";

    detections.forEach(det => {
        const item = document.createElement("div");
        item.className = "result-item";
        item.setAttribute("role", "listitem");
        const color = CLASS_COLORS[det.type] || "#fff";
        item.innerHTML = `
            <span class="color-dot" style="background:${color}"></span>
            <span class="label">${det.label}</span>
            <span class="confidence">${(det.confidence * 100).toFixed(1)}%</span>
        `;
        resultsList.appendChild(item);
    });
}

// --- Live Detection Mode ---
liveConfSlider.addEventListener("input", () => {
    liveConfValue.textContent = liveConfSlider.value;
});

fpsLimitSlider.addEventListener("input", () => {
    fpsLimitValue.textContent = fpsLimitSlider.value;
});

startLiveBtn.addEventListener("click", startLiveDetection);
stopLiveBtn.addEventListener("click", stopLiveDetection);

async function startLiveDetection() {
    if (!detector.session) {
        setStatus("Model still loading, please wait...", "error");
        return;
    }

    try {
        liveStream = await navigator.mediaDevices.getDisplayMedia({
            video: { cursor: "always" },
            audio: false
        });
    } catch (err) {
        setStatus("Screen capture denied or not supported: " + err.message, "error");
        return;
    }

    liveVideo.srcObject = liveStream;
    await liveVideo.play();

    liveRunning = true;
    startLiveBtn.classList.add("hidden");
    stopLiveBtn.classList.remove("hidden");
    liveIndicator.classList.remove("hidden");

    liveOverlay.width = liveVideo.videoWidth;
    liveOverlay.height = liveVideo.videoHeight;

    setStatus("Live detection active. Detecting UI elements in real-time...", "success");

    liveStream.getVideoTracks()[0].addEventListener("ended", stopLiveDetection);

    lastFrameTime = performance.now();
    fpsUpdateTime = performance.now();
    frameCount = 0;
    processVideoFrame();
}

function stopLiveDetection() {
    liveRunning = false;

    if (liveAnimFrame) {
        cancelAnimationFrame(liveAnimFrame);
        liveAnimFrame = null;
    }

    if (liveStream) {
        liveStream.getTracks().forEach(t => t.stop());
        liveStream = null;
    }

    liveVideo.srcObject = null;
    const ctx = liveOverlay.getContext("2d");
    ctx.clearRect(0, 0, liveOverlay.width, liveOverlay.height);

    startLiveBtn.classList.remove("hidden");
    stopLiveBtn.classList.add("hidden");
    liveIndicator.classList.add("hidden");
    fpsCounter.textContent = "0 FPS";

    liveResultsList.innerHTML = "";
    liveCount.textContent = "(0)";
    liveResults.classList.add("hidden");

    setStatus("Live detection stopped.", "");
}

async function processVideoFrame() {
    if (!liveRunning) return;

    liveAnimFrame = requestAnimationFrame(processVideoFrame);

    const now = performance.now();
    const fpsLimit = parseInt(fpsLimitSlider.value, 10);
    const minFrameMs = 1000 / fpsLimit;

    if (now - lastFrameTime < minFrameMs) return;
    lastFrameTime = now;

    // FPS counter
    frameCount++;
    if (now - fpsUpdateTime >= 1000) {
        currentFps = frameCount;
        fpsCounter.textContent = currentFps + " FPS";
        frameCount = 0;
        fpsUpdateTime = now;
    }

    if (!liveVideo.videoWidth || !liveVideo.videoHeight) return;

    try {
        detector.confThreshold = parseFloat(liveConfSlider.value);

        const tempCanvas = document.createElement("canvas");
        tempCanvas.width = liveVideo.videoWidth;
        tempCanvas.height = liveVideo.videoHeight;
        const tempCtx = tempCanvas.getContext("2d");
        tempCtx.drawImage(liveVideo, 0, 0);

        const detections = await detector.detect(tempCanvas);
        liveDetections = YoloVisionAdapter.adaptDetections(detections);

        drawLiveOverlay(liveDetections);
        renderLiveResults(liveDetections);
    } catch (err) {
        console.warn("Live detection frame error:", err);
    }
}

function drawLiveOverlay(detections) {
    const ctx = liveOverlay.getContext("2d");
    const w = liveOverlay.width;
    const h = liveOverlay.height;
    ctx.clearRect(0, 0, w, h);

    detections.forEach(det => {
        const { bbox, type, confidence } = det;
        const color = CLASS_COLORS[type] || "#fff";

        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.strokeRect(bbox.x1, bbox.y1, bbox.x2 - bbox.x1, bbox.y2 - bbox.y1);

        const label = `${type} ${(confidence * 100).toFixed(0)}%`;
        ctx.font = "bold 14px sans-serif";
        const tw = ctx.measureText(label).width;

        ctx.fillStyle = color;
        ctx.fillRect(bbox.x1, bbox.y1 - 22, tw + 12, 22);

        ctx.fillStyle = "#fff";
        ctx.fillText(label, bbox.x1 + 6, bbox.y1 - 6);
    });
}

function renderLiveResults(detections) {
    liveCount.textContent = `(${detections.length})`;

    if (detections.length === 0) {
        liveResults.classList.add("hidden");
        return;
    }

    liveResults.classList.remove("hidden");
    liveResultsList.innerHTML = "";

    const grouped = {};
    detections.forEach(det => {
        if (!grouped[det.type]) grouped[det.type] = [];
        grouped[det.type].push(det);
    });

    Object.entries(grouped).forEach(([cls, dets]) => {
        dets.forEach(det => {
            const item = document.createElement("div");
            item.className = "result-item";
            const color = CLASS_COLORS[det.type] || "#fff";
            item.innerHTML = `
                <span class="color-dot" style="background:${color}"></span>
                <span class="label">${det.label}</span>
                <span class="confidence">${(det.confidence * 100).toFixed(1)}%</span>
            `;
            liveResultsList.appendChild(item);
        });
    });
}

// --- Init ---
initModel();
