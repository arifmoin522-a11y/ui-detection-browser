# UI Element Detection - Browser

> Lightweight browser-based UI element detection using [foduucom/web-form-ui-field-detection](https://huggingface.co/foduucom/web-form-ui-field-detection) (YOLOv8s) running entirely client-side via ONNX Runtime Web.

Detect buttons, inputs, dropdowns, checkboxes, radio buttons, text fields, and 39 other UI element types from screenshots - **no server required**.

---

## Demo

```
┌─────────────────────────────────────────────┐
│  [Upload Screenshot]  [Detect]  Conf: 0.25  │
├─────────────────────────────────────────────┤
│                                             │
│    ┌──────┐   ┌─────────────────┐          │
│    │button│   │   text input    │          │
│    └──────┘   └─────────────────┘          │
│                                             │
│    ┌────────┐  ┌─────┐  ┌──────────┐      │
│    │ submit │  │radio│  │ checkbox │      │
│    └────────┘  └─────┘  └──────────┘      │
│                                             │
├─────────────────────────────────────────────┤
│  Detected: 4 elements                       │
│  ● button 95.2%  ● text 88.7%              │
│  ● radio 91.3%   ● checkbox 87.1%          │
└─────────────────────────────────────────────┘
```

---

## How It Works

```mermaid
flowchart TD
    A[User Uploads Screenshot] --> B[Image Preprocessing]
    B --> C[ONNX Runtime Web]
    C --> D[YOLOv8s Inference]
    D --> E[Postprocessing - NMS]
    E --> F[Bounding Box Rendering]
    F --> G[Results Display]

    subgraph "Browser (Client-Side)"
        B
        C
        D
        E
        F
        G
    end

    subgraph "Model"
        D
    end

    H[web-form-ui-field-detection] -.-> D
```

---

## Architecture

```mermaid
flowchart LR
    subgraph Input
        A[Screenshot/Image]
    end

    subgraph Preprocessing
        B[Resize to 640x640]
        C[Normalize 0-1]
        D[CHW Layout]
    end

    subgraph YOLOv8s
        E[Backbone]
        F[Neck - PANet]
        G[Detection Head]
    end

    subgraph Postprocessing
        H[Confidence Filter]
        I[NMS - IoU 0.45]
        J[Coordinate Transform]
    end

    subgraph Output
        K[Bounding Boxes]
        L[Class Labels]
        M[Confidence Scores]
    end

    A --> B --> C --> D --> E --> F --> G --> H --> I --> J --> K & L & M
```

---

## Model Details

| Property | Value |
|----------|-------|
| Architecture | YOLOv8s (Small) |
| Task | Object Detection |
| Input Size | 640x640 RGB |
| Classes | 39 UI element types |
| Precision | 0.80 |
| Recall | 0.70 |
| F1 Score | 0.71 |
| mAP@0.5 | 0.51 |
| Model Size | ~22 MB |
| License | Open |

### Detected Classes

```
button, checkbox, combobox, date, dropdown, email, file, image, link,
listbox, number, password, radio, range, reset, search, submit, tel,
text, textarea, time, url, username, address, city, country,
credit-card, cvv, expiration, first-name, last-name, phone, pin,
postal-code, state, message, comment, search-box
```

---

## Project Structure

```
ui-detection-browser/
├── README.md
├── package.json
├── requirements.txt
├── scripts/
│   └── export_model.py          # Export YOLOv8 to ONNX/TFJS
├── model/
│   └── ui-detection.onnx        # Exported ONNX model (after running export)
├── public/
│   ├── index.html               # Browser demo UI
│   ├── css/
│   │   └── style.css
│   ├── js/
│   │   ├── detector.js          # ONNX inference engine
│   │   └── app.js               # UI controller
│   └── server.py                # Local HTTP server
├── docs/
│   └── export.ipynb             # Jupyter notebook for model export
└── .gitignore
```

---

## Quick Start

### 1. Export Model to ONNX

```bash
pip install -r requirements.txt
python scripts/export_model.py onnx
```

This downloads the model from Hugging Face and exports it as `model/ui-detection.onnx`.

### 2. Run in Browser

```bash
cd public
python server.py
# Open http://localhost:8000
```

Or use any static file server:

```bash
npx serve public
```

### 3. Upload a screenshot and click Detect

---

## Export Formats

### ONNX (Recommended for Browser)

```bash
python scripts/export_model.py onnx
```

Uses [ONNX Runtime Web](https://onnxruntime.ai/docs/tutorials/web/) with WebGL/WASM backends.

### TensorFlow.js

```bash
python scripts/export_model.py tfjs
```

Uses [TensorFlow.js](https://www.tensorflow.org/js) with WebGPU/WebGL backends.

---

## How the Pipeline Works

```mermaid
sequenceDiagram
    participant U as User
    participant A as App (JS)
    participant O as ONNX Runtime
    participant M as YOLOv8s Model

    U->>A: Upload Screenshot
    A->>A: Read image file
    A->>A: Resize to 640x640
    A->>A: Normalize pixels (0-1)
    A->>A: Rearrange to CHW layout
    A->>O: Create Tensor
    O->>M: Run Inference
    M-->>O: Raw predictions [1, 43, 8400]
    O-->>A: Output tensor
    A->>A: Filter by confidence > 0.25
    A->>A: Apply NMS (IoU < 0.45)
    A->>A: Map coordinates to original size
    A->>U: Draw bounding boxes on canvas
    A->>U: Show detected elements list
```

---

## Supported UI Elements

```mermaid
mindmap
  root((UI Elements))
    Form Inputs
      text
      email
      password
      number
      tel
      url
      search
      date
      time
    Buttons
      button
      submit
      reset
    Selection
      checkbox
      radio
      dropdown
      combobox
      listbox
      range
    Media
      image
      file
    Navigation
      link
      search-box
    Labels
      message
      comment
    Addresses
      address
      city
      state
      country
      postal-code
    Identity
      first-name
      last-name
      username
      phone
      pin
    Payment
      credit-card
      cvv
      expiration
```

---

## Model Performance

```mermaid
xychart-beta
    title "Model Performance Metrics"
    x-axis ["Precision", "Recall", "F1", "mAP@0.5"]
    y-axis "Score" 0 --> 1
    bar [0.80, 0.70, 0.71, 0.51]
```

---

## Training Details

| Parameter | Value |
|-----------|-------|
| Base Model | YOLOv8s pretrained |
| Dataset | 600 annotated web form images |
| Optimizer | Adam (lr=1e-4) |
| Loss | mAP loss |
| GPU | NVIDIA RTX 3090 |
| Training Time | ~1 hour |

---

## Browser Compatibility

| Browser | Backend | Status |
|---------|---------|--------|
| Chrome 90+ | WebGL2 | Supported |
| Firefox 90+ | WebGL2 | Supported |
| Safari 15+ | WebGL2 | Supported |
| Edge 90+ | WebGL2 | Supported |

---

## Credits

- Model: [foduucom/web-form-ui-field-detection](https://huggingface.co/foduucom/web-form-ui-field-detection)
- Inference: [ONNX Runtime Web](https://onnxruntime.ai/)
- Architecture: [YOLOv8](https://github.com/ultralytics/ultralytics)

---

## License

MIT
