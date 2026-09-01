/**
 * YOLO Vision Adapter
 * Converts YOLO detector output to team contract format.
 *
 * Input:  { class, classIdx, score, bbox: {x1,y1,x2,y2} }
 * Output: { id, type, label, bbox: {x1,y1,x2,y2}, confidence }
 */

const YoloVisionAdapter = {
    adaptDetections(detections) {
        return detections.map((det, index) => ({
            id: index,
            type: det.class,
            label: det.class,
            bbox: det.bbox,
            confidence: det.score
        }));
    }
};
