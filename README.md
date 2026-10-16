# License Plate Recognition & Vehicle Classification

Detects license plates and classifies the vehicle type (car / truck / motorcycle) with a custom-trained **YOLO** model, then reads the plate text with **PaddleOCR**. Runs entirely on **CPU**, as a live-camera script or a **Gradio** web app.

> Developed and tested on Windows with Python 3.13, CPU only.

## Features

- **One model, four classes:** `License_Plate`, `cars`, `motorcyle`, `truck`. Every box is drawn with its class name; OCR runs only on plate boxes.
- **Two interfaces:** a live-camera script (`plate_02.py`) and a web app (`app_02.py`) for image upload or webcam.
- **Robust plate reading:**
  - merges multi-line plates (for example Jordanian plates: two digits on top, five below)
  - filters low-confidence noise and stray text such as phone numbers and country names (fuzzy matching with `difflib`)
  - supports alphanumeric plates (for example UK plates), not just digits
  - stabilizes live output with a majority vote over the last 5 readings
- **Built for modest hardware:** frame skipping, scheduled OCR, lightweight PaddleOCR settings, no GPU needed.

## How it works

```
frame / image
   |
   v
YOLO detector (4 classes)
   |-- cars / truck / motorcyle  -->  draw box + class name
   |
   '-- License_Plate  -->  crop  -->  grayscale, 3x upscale, CLAHE
                                          |
                                          v
                                      PaddleOCR
                                          |
                                          v
              filter (confidence, length, country name)  -->  merge segments
                                          |
                                          v
                         majority vote (live mode)  -->  plate text
```

## Project structure

```
.
├── app_02.py            # Gradio web app (single image per request)
├── plate_02.py          # live-camera script
├── plate_utils_01.py    # shared logic: cleaning, validation, preprocessing,
│                        #   model loading, read_plate()
├── models/
│   ├── best.pt          # trained YOLO weights (4 classes, ~6 MB)
│   └── README.md
├── docs/
│   └── project-report-ar.docx   # project report (Arabic)
├── requirements.txt
└── README.md
```

## Installation

```bash
git clone https://github.com/Mou9tafa/<repo-name>.git
cd <repo-name>

python -m venv .venv
# Windows PowerShell:  .venv\Scripts\Activate.ps1
# Linux / macOS:       source .venv/bin/activate

pip install -r requirements.txt
```

The trained weights are included at `models/best.pt`. To use your own, replace that file or set `PLATE_MODEL_PATH` (see `models/README.md`).

## Usage

**Web app**

```bash
python app_02.py
```

Open the local URL printed in the terminal (usually http://127.0.0.1:7860), then upload an image or use the webcam. The result is the annotated image plus the text of each detected plate with its confidence.

**Live camera**

```bash
python plate_02.py
```

Press `q` in the video window to quit.

## Configuration

| Setting | File | Default | Purpose |
|---|---|---|---|
| `yolo_conf` | `app_02.py`, `plate_02.py` | `0.25` | Minimum YOLO confidence (low on purpose, so weak vehicle detections stay visible) |
| `skip_frames` | `plate_02.py` | `3` | Run YOLO on 1 of every 3 frames |
| `ocr_interval` | `plate_02.py` | `10` | OCR runs when the frame counter is a multiple of this. Because YOLO only runs on every 3rd frame, OCR effectively runs every 30 frames |
| `MIN_OCR_CONFIDENCE` | `plate_utils_01.py` | `0.75` | Minimum confidence for an OCR text segment |
| `MAX_SEGMENT_LENGTH` | `plate_utils_01.py` | `7` | Longest accepted single segment (drops phone numbers and ad text) |
| `IGNORED_PLATE_WORDS` | `plate_utils_01.py` | `jordan, jo, ...` | Country-name text to ignore (matched fuzzily) |
| `DEBUG`, `OCR_DEBUG` | scripts / `plate_utils_01.py` | `False` | Print raw detections and OCR segments to the terminal |

The YOLO input size is set to `imgsz=640` in both scripts; small or distant plates are missed at lower sizes.

## Model and dataset

The YOLO model was trained on a dataset from [Roboflow Universe](https://universe.roboflow.com/) with the classes `License_Plate`, `cars`, `motorcyle`, `truck`.

Dataset: **[add dataset name, link and license here]**

## Known limitations

- Vehicle-type detection is weaker than plate detection (a truck was missed in testing, and a car in a distant shot scored low).
- No formal benchmark yet: thresholds were tuned on a small set of real photos, so they may not generalize.
- With the low default `yolo_conf`, a weak duplicate box can appear on the same plate and its label may overlap the main one. Raising `yolo_conf` removes it but also hides weak vehicle detections.
- The OCR model is English-only: Arabic text on the plate is not recognized (it is ignored by design).
- The fuzzy country-name filter can wrongly drop very short look-alikes of `JO` (for example `JOB`).
- In live mode the last stable text stays on screen until a new one stabilizes; there is no tracking, so two plates in one frame share one label.
- In single-image mode there is no voting, so one dropped digit is not corrected.

## Roadmap

- Build a labeled test set and report detection precision/recall and plate-level accuracy.
- Add a pretrained general detector (for example a COCO YOLO) for car / truck / bus / motorcycle, or retrain with more vehicle images.
- Add tracking so each plate keeps its own stable text.
- Add an Arabic or bilingual OCR model.
- Export to ONNX/OpenVINO for faster CPU inference.
- Use a separate, stricter confidence threshold for plate boxes to remove duplicate detections.
- Replace word-list filtering with per-country plate-format rules.

## Acknowledgments

[Ultralytics YOLO](https://github.com/ultralytics/ultralytics), [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR), [OpenCV](https://opencv.org/), [Gradio](https://www.gradio.app/), and the dataset authors on Roboflow Universe.

## Author

Moustafa · [GitHub](https://github.com/Mou9tafa)
