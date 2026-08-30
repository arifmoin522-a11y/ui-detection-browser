/**
 * Application entry point
 * Handles UI interactions and orchestrates detection
 */

const imageInput = document.getElementById("imageInput");
const detectBtn = document.getElementById("detectBtn");
const statusEl = document.getElementById("status");
const canvas = document.getElementById("outputCanvas");
const loadingOverlay = document.getElementById("loadingOverlay");
const resultsEl = document.getElementById("results");
const resultsList = document.getElementById("resultsList");
const confSlider = document.getElementById("confThreshold");
const confValue = document.getElementById("confValue");

let currentImage = null;
const detector = new UIDetector();

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

        renderResults(detections);

        setStatus(`Found ${detections.length} UI element(s)`, "success");
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

    const grouped = {};
    detections.forEach(det => {
        if (!grouped[det.class]) grouped[det.class] = [];
        grouped[det.class].push(det);
    });

    Object.entries(grouped).forEach(([cls, dets]) => {
        dets.forEach(det => {
            const item = document.createElement("div");
            item.className = "result-item";

            const color = CLASS_COLORS[det.class] || "#fff";
            item.innerHTML = `
                <span class="color-dot" style="background:${color}"></span>
                <span class="label">${det.class}</span>
                <span class="confidence">${(det.score * 100).toFixed(1)}%</span>
            `;
            resultsList.appendChild(item);
        });
    });
}

initModel();
