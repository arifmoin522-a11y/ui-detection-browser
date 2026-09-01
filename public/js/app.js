/**
 * Application entry point
 * Handles UI interactions and orchestrates detection
 */

const imageInput = document.getElementById("imageInput");
const detectBtn = document.getElementById("detectBtn");
const resetBtn = document.getElementById("resetBtn");
const statusEl = document.getElementById("status");
const canvas = document.getElementById("outputCanvas");
const loadingOverlay = document.getElementById("loadingOverlay");
const resultsEl = document.getElementById("results");
const resultsList = document.getElementById("resultsList");
const confSlider = document.getElementById("confThreshold");
const confValue = document.getElementById("confValue");

let currentImage = null;
let detectionHistory = [];
const detector = new UIDetector();

resetBtn.addEventListener("click", resetView);

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
    resetBtn.style.display = "none";
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
        resetBtn.style.display = "none";

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
        resetBtn.style.display = "inline-block";

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
        resetBtn.style.display = "none";
        return;
    }

    resultsEl.classList.remove("hidden");
    resultsList.innerHTML = "";

    const grouped = {};
    detections.forEach(det => {
        if (!grouped[det.type]) grouped[det.type] = [];
        grouped[det.type].push(det);
    });

    Object.entries(grouped).forEach(([cls, dets]) => {
        dets.forEach(det => {
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
    });
}

initModel();