/**
 * UI Element Detector using ONNX Runtime Web
 * Handles model loading, preprocessing, inference, and postprocessing
 */

const CLASSES = [
    "DOB", "address", "age input", "age", "button", "checkbox",
    "city", "company", "country dropdown", "country input", "date",
    "day dropdown", "doc-upload", "dropdown", "email-input", "emp id",
    "first-name", "gender dropdown", "gender", "input", "job role",
    "last-name", "message", "month dropdown", "name", "otp",
    "password", "phone-num", "redio button", "region",
    "reminder checkbox", "state dropdown", "state input-", "state",
    "terms checkbox", "username", "web url-", "year dropdown", "zip code"
];

const CLASS_COLORS = {};
CLASSES.forEach((cls, i) => {
    const hue = (i * 360 / CLASSES.length) % 360;
    CLASS_COLORS[cls] = `hsl(${hue}, 70%, 55%)`;
});

class UIDetector {
    constructor(modelPath = "model/ui-detection.onnx") {
        this.modelPath = modelPath;
        this.session = null;
        this.inputShape = [1, 3, 640, 640];
        this.confThreshold = 0.25;
        this.iouThreshold = 0.45;
    }

    async loadModel() {
        this.session = await ort.InferenceSession.create(this.modelPath, {
            executionProviders: ["webgl", "wasm"],
            graphOptimizationLevel: "all"
        });
        return true;
    }

    preprocess(imageElement) {
        const canvas = document.createElement("canvas");
        const [_, channels, height, width] = this.inputShape;
        canvas.width = width;
        canvas.height = height;
        const ctx = canvas.getContext("2d");

        const scale = Math.min(width / imageElement.width, height / imageElement.height);
        const offsetX = (width - imageElement.width * scale) / 2;
        const offsetY = (height - imageElement.height * scale) / 2;

        ctx.fillStyle = "#000";
        ctx.fillRect(0, 0, width, height);
        ctx.drawImage(imageElement, offsetX, offsetY, imageElement.width * scale, imageElement.height * scale);

        const imageData = ctx.getImageData(0, 0, width, height);
        const pixels = new Uint32Array(imageData.data.buffer);
        const channelSize = height * width;
        const float32Data = new Float32Array(channels * channelSize);

        for (let i = 0; i < channelSize; i++) {
            const p = pixels[i];
            float32Data[i] = (p & 0xff) / 255.0;                       // R
            float32Data[channelSize + i] = ((p >> 8) & 0xff) / 255.0;   // G
            float32Data[2 * channelSize + i] = ((p >> 16) & 0xff) / 255.0; // B
        }

        return {
            tensor: new ort.Tensor("float32", float32Data, this.inputShape),
            scale,
            offsetX,
            offsetY,
            origWidth: imageElement.width,
            origHeight: imageElement.height
        };
    }

    async detect(imageElement) {
        if (!this.session) {
            throw new Error("Model not loaded. Call loadModel() first.");
        }

        const { tensor, scale, offsetX, offsetY, origWidth, origHeight } = this.preprocess(imageElement);

        const inputName = this.session.inputNames[0];
        const feeds = { [inputName]: tensor };
        const results = await this.session.run(feeds);

        const outputName = this.session.outputNames[0];
        const output = results[outputName];

        return this.postprocess(output.data, scale, offsetX, offsetY, origWidth, origHeight);
    }

    postprocess(data, scale, offsetX, offsetY, origWidth, origHeight) {
        const detections = [];
        const [_, __, H, W] = this.inputShape;

        const numDetections = data.length / (CLASSES.length + 4);

        for (let i = 0; i < numDetections; i++) {
            const base = i * (CLASSES.length + 4);
            const cx = data[base];
            const cy = data[base + 1];
            const w = data[base + 2];
            const h = data[base + 3];

            let maxScore = 0;
            let maxClassIdx = 0;
            for (let j = 0; j < CLASSES.length; j++) {
                const score = data[base + 4 + j];
                if (score > maxScore) {
                    maxScore = score;
                    maxClassIdx = j;
                }
            }

            if (maxScore < this.confThreshold) continue;

            const x1 = (cx - w / 2 - offsetX) / scale;
            const y1 = (cy - h / 2 - offsetY) / scale;
            const x2 = (cx + w / 2 - offsetX) / scale;
            const y2 = (cy + h / 2 - offsetY) / scale;

            detections.push({
                class: CLASSES[maxClassIdx],
                classIdx: maxClassIdx,
                score: maxScore,
                bbox: {
                    x1: Math.max(0, x1),
                    y1: Math.max(0, y1),
                    x2: Math.min(origWidth, x2),
                    y2: Math.min(origHeight, y2)
                }
            });
        }

        return this.nms(detections);
    }

    nms(detections) {
        detections.sort((a, b) => b.score - a.score);
        const n = detections.length;
        const suppressed = new Uint8Array(n);

        for (let i = 0; i < n; i++) {
            if (suppressed[i]) continue;
            const best = detections[i];
            for (let j = i + 1; j < n; j++) {
                if (suppressed[j]) continue;
                if (detections[j].class !== best.class) continue;
                if (this.computeIoU(best.bbox, detections[j].bbox) >= this.iouThreshold) {
                    suppressed[j] = 1;
                }
            }
        }

        const kept = [];
        for (let i = 0; i < n; i++) {
            if (!suppressed[i]) kept.push(detections[i]);
        }
        return kept;
    }

    computeIoU(a, b) {
        const x1 = Math.max(a.x1, b.x1);
        const y1 = Math.max(a.y1, b.y1);
        const x2 = Math.min(a.x2, b.x2);
        const y2 = Math.min(a.y2, b.y2);

        const intersection = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
        const areaA = (a.x2 - a.x1) * (a.y2 - a.y1);
        const areaB = (b.x2 - b.x1) * (b.y2 - b.y1);

        return intersection / (areaA + areaB - intersection + 1e-6);
    }

    drawDetections(canvas, imageElement, detections) {
        const ctx = canvas.getContext("2d");
        canvas.width = imageElement.width;
        canvas.height = imageElement.height;
        ctx.drawImage(imageElement, 0, 0);

        detections.forEach(det => {
            const { bbox, class: cls, score } = det;
            const color = CLASS_COLORS[cls] || "#fff";

            ctx.strokeStyle = color;
            ctx.lineWidth = 2;
            ctx.strokeRect(bbox.x1, bbox.y1, bbox.x2 - bbox.x1, bbox.y2 - bbox.y1);

            const text = `${cls} ${(score * 100).toFixed(0)}%`;
            ctx.font = "bold 14px sans-serif";
            const textWidth = ctx.measureText(text).width;

            ctx.fillStyle = color;
            ctx.fillRect(bbox.x1, bbox.y1 - 22, textWidth + 12, 22);

            ctx.fillStyle = "#fff";
            ctx.fillText(text, bbox.x1 + 6, bbox.y1 - 6);
        });
    }
}
