/**
 * YoloVisionAdapter Tests
 * Run in browser console or with a testing framework.
 *
 * Usage:
 *   Run in browser console: YoloVisionAdapterTestSuite.run()
 */

const YoloVisionAdapterTestSuite = {
    results: { passed: 0, failed: 0, total: 0 },

    assert(condition, message) {
        this.total++;
        if (condition) {
            this.passed++;
            console.log(`  [PASS] ${message}`);
        } else {
            this.failed++;
            console.error(`  [FAIL] ${message}`);
        }
    },

    run() {
        console.log("=== YoloVisionAdapter Test Suite ===\n");

        this.testBasicAdaptation();
        this.testEmptyInput();
        this.testInvalidInput();
        this.testInvalidBbox();
        this.testInvalidScore();
        this.testUniqueIds();
        this.testSorting();
        this.testFieldMapping();
        this.testMultipleCalls();

        console.log(`\n=== Results: ${this.passed}/${this.total} passed, ${this.failed} failed ===`);
        return this.failed === 0;
    },

    testBasicAdaptation() {
        console.log("\n[1] Basic Adaptation");
        const input = [
            { class: "button", classIdx: 0, score: 0.95, bbox: { x1: 10, y1: 20, x2: 100, y2: 50 } }
        ];
        const output = YoloVisionAdapter.adaptDetections(input);
        this.assert(output.length === 1, "Output has 1 detection");
        this.assert(output[0].id === "ui-1", `ID is ui-1, got ${output[0].id}`);
        this.assert(output[0].type === "button", `Type is button, got ${output[0].type}`);
        this.assert(output[0].label === "button", `Label is button, got ${output[0].label}`);
        this.assert(output[0].confidence === 0.95, `Confidence is 0.95, got ${output[0].confidence}`);
        this.assert(JSON.stringify(output[0].bbox) === JSON.stringify({ x1: 10, y1: 20, x2: 100, y2: 50 }),
            "Bbox preserved correctly");
        this.assert(output[0].bbox !== input[0].bbox, "Bbox is a copy (immutability)");
    },

    testEmptyInput() {
        console.log("\n[2] Empty Input");
        const output = YoloVisionAdapter.adaptDetections([]);
        this.assert(output.length === 0, "Empty input returns empty array");
    },

    testInvalidInput() {
        console.log("\n[3] Invalid Input Types");
        const output1 = YoloVisionAdapter.adaptDetections(null);
        this.assert(output1.length === 0, "null returns empty array");
        const output2 = YoloVisionAdapter.adaptDetections("string");
        this.assert(output2.length === 0, "string returns empty array");
        const output3 = YoloVisionAdapter.adaptDetections(undefined);
        this.assert(output3.length === 0, "undefined returns empty array");
    },

    testInvalidBbox() {
        console.log("\n[4] Invalid Bounding Box");
        const input = [
            { class: "button", score: 0.95, bbox: { x1: 10, y1: 20, x2: 5, y2: 50 } },
            { class: "input", score: 0.9, bbox: { x1: 10, y1: 20 } }
        ];
        const output = YoloVisionAdapter.adaptDetections(input);
        this.assert(output.length === 0, "Invalid bbox filtered out");
    },

    testInvalidScore() {
        console.log("\n[5] Invalid Score");
        const input = [
            { class: "button", score: 1.5, bbox: { x1: 10, y1: 20, x2: 100, y2: 50 } },
            { class: "input", score: -0.1, bbox: { x1: 10, y1: 20, x2: 100, y2: 50 } }
        ];
        const output = YoloVisionAdapter.adaptDetections(input);
        this.assert(output.length === 0, "Invalid scores filtered out");
    },

    testUniqueIds() {
        console.log("\n[6] Unique IDs");
        const input = [
            { class: "a", score: 0.5, bbox: { x1: 0, y1: 0, x2: 10, y2: 10 } },
            { class: "b", score: 0.5, bbox: { x1: 0, y1: 0, x2: 10, y2: 10 } },
            { class: "c", score: 0.5, bbox: { x1: 0, y1: 0, x2: 10, y2: 10 } }
        ];
        const output = YoloVisionAdapter.adaptDetections(input);
        const ids = output.map(d => d.id);
        const unique = new Set(ids);
        this.assert(unique.size === 3, "All IDs are unique");
        this.assert(ids[0] === "ui-1" && ids[1] === "ui-2" && ids[2] === "ui-3", "IDs are sequential");
    },

    testSorting() {
        console.log("\n[7] Sort by Confidence (descending)");
        const input = [
            { class: "low", score: 0.1, bbox: { x1: 0, y1: 0, x2: 10, y2: 10 } },
            { class: "high", score: 0.9, bbox: { x1: 0, y1: 0, x2: 10, y2: 10 } },
            { class: "mid", score: 0.5, bbox: { x1: 0, y1: 0, x2: 10, y2: 10 } }
        ];
        const output = YoloVisionAdapter.adaptDetections(input);
        this.assert(output[0].confidence === 0.9, "Highest confidence first");
        this.assert(output[1].confidence === 0.5, "Mid confidence second");
        this.assert(output[2].confidence === 0.1, "Lowest confidence last");
    },

    testFieldMapping() {
        console.log("\n[8] Field Mapping");
        const input = [{ class: "submit", classIdx: 5, score: 0.88, bbox: { x1: 5, y1: 10, x2: 200, y2: 40 } }];
        const output = YoloVisionAdapter.adaptDetections(input);
        this.assert(output[0].type === "submit", "type field mapped from class");
        this.assert(output[0].label === "submit", "label field mapped from class");
        this.assert(output[0].confidence === 0.88, "confidence field mapped from score");
        this.assert(!("class" in output[0]), "Original class field not in output");
        this.assert(!("score" in output[0]), "Original score field not in output");
        this.assert(!("classIdx" in output[0]), "Original classIdx field not in output");
    },

    testMultipleCalls() {
        console.log("\n[9] Multiple Calls");
        let threw = false;
        try {
            for (let i = 0; i < 5; i++) {
                const input = [{ class: `cls${i}`, score: 0.5, bbox: { x1: 0, y1: 0, x2: 10, y2: 10 } }];
                YoloVisionAdapter.adaptDetections(input);
            }
        } catch (e) {
            threw = true;
        }
        this.assert(!threw, "Multiple calls do not throw");
    }
};
