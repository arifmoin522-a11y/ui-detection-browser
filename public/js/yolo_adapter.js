/**
 * YOLO Vision Adapter
 * Converts YOLO detector output to team contract format.
 *
 * Input:  { class, classIdx, score, bbox: {x1,y1,x2,y2} }
 * Output: { id, type, label, bbox: {x1,y1,x2,y2}, confidence }
 *
 * Usage:
 *   const visionOutput = YoloVisionAdapter.adaptDetections(detections);
 */

let _idCounter = 0;

const VALID_BBOX = (bbox) =>
    bbox &&
    typeof bbox.x1 === "number" &&
    typeof bbox.y1 === "number" &&
    typeof bbox.x2 === "number" &&
    typeof bbox.y2 === "number" &&
    bbox.x2 >= bbox.x1 &&
    bbox.y2 >= bbox.y1;

const VALID_SCORE = (score) =>
    typeof score === "number" && score >= 0 && score <= 1;

const YoloVisionAdapter = {
    adaptDetections(detections) {
        if (!Array.isArray(detections)) {
            console.warn("[YoloVisionAdapter] Expected array, got:", typeof detections);
            return [];
        }

        _idCounter = 0;

        return detections
            .filter((det) => {
                if (!det || typeof det !== "object") {
                    console.warn("[YoloVisionAdapter] Skipping invalid detection:", det);
                    return false;
                }
                if (!VALID_BBOX(det.bbox)) {
                    console.warn("[YoloVisionAdapter] Skipping detection with invalid bbox:", det.bbox);
                    return false;
                }
                if (!VALID_SCORE(det.score)) {
                    console.warn("[YoloVisionAdapter] Skipping detection with invalid score:", det.score);
                    return false;
                }
                return true;
            })
            .map((det) => ({
                id: `ui-${++_idCounter}`,
                type: det.class || det.type || "unknown",
                label: det.class || det.type || "unknown",
                bbox: { ...det.bbox },
                confidence: det.score,
            }))
            .sort((a, b) => b.confidence - a.confidence);
    },

    resetCounter() {
        _idCounter = 0;
    }
};
