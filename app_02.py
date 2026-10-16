import os

# =========================================================
# PaddlePaddle Settings
# IMPORTANT: Must be BEFORE importing PaddleOCR
# =========================================================

os.environ["FLAGS_enable_pir_api"] = "0"            # turn off Paddle's newer PIR execution engine (unstable on some setups)
os.environ["FLAGS_use_mkldnn"] = "0"                # turn off Intel MKL-DNN acceleration (can crash on non-Intel / older CPUs)
os.environ["FLAGS_use_onednn"] = "0"                # turn off oneDNN acceleration (same reason as above)
os.environ["PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT"] = "0"  # same MKL-DNN toggle, but for the PaddleX interface

import sys
import cv2
import gradio as gr

from plate_utils_01 import load_yolo_model, load_ocr_model, read_plate, is_plate_class


# =========================================================
# 1. Load Models (with clear error messages)
# =========================================================

print("Loading YOLO model...")

try:
    model = load_yolo_model()
except FileNotFoundError as e:
    print(f"ERROR: {e}")
    sys.exit(1)

print("Loading License Plate Recognition model...")

try:
    ocr = load_ocr_model()
except Exception as e:
    print(f"ERROR: Failed to load PaddleOCR: {e}")
    sys.exit(1)

print("Models loaded successfully \u2705")
print("Model classes:", model.names)

# Lowered from 0.5 so weaker detections (smaller/farther plates)
# aren't silently dropped before you can even see them.
yolo_conf = 0.25

# Prints every detected box's class + confidence to the console.
# Set to False once you've confirmed detection is working well.
DEBUG = False
print("Model classes:", model.names)


# =========================================================
# 2. Main Processing Function
# =========================================================

def process_plate(image):

    if image is None:
        return None, "لم يتم إدخال صورة."

    # Gradio gives us an RGB numpy array. OpenCV drawing colors
    # (cv2.rectangle / cv2.putText) are conventionally BGR, so we
    # draw on a BGR copy and convert back to RGB at the end.
    # (Drawing directly on the RGB array swapped red/blue in the
    # original version.)
    annotated_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    detected_plates = []

    # -------------------------------------------------------
    # YOLO Detection
    # -------------------------------------------------------

    try:
        results = model(image, imgsz=640, conf=yolo_conf, verbose=False)
    except Exception as e:
        print("YOLO Error:", e)
        return image, f"حدث خطأ في YOLO: {e}"

    for r in results:

        if r.boxes is None:
            continue

        boxes = r.boxes.xyxy.cpu().numpy()
        class_ids = r.boxes.cls.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()

        for box, cls_id, conf in zip(boxes, class_ids, confs):

            x1, y1, x2, y2 = map(int, box)

            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(image.shape[1], x2)
            y2 = min(image.shape[0], y2)

            class_name = model.names[int(cls_id)]

            if DEBUG:
                print(f"[DEBUG] class={class_name}  conf={conf:.2f}")

            crop = image[y1:y2, x1:x2]

            if crop.size == 0:
                continue

            # =================================================
            # Draw Detection + Class Label (on the BGR copy)
            # Works for both vehicle-type and plate boxes.
            # =================================================

            cv2.rectangle(annotated_bgr, (x1, y1), (x2, y2), (255, 0, 0), 2)

            label_y = y1 - 10

            if label_y < 20:
                label_y = y1 + 20

            cv2.putText(
                annotated_bgr, class_name, (x1, label_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2
            )

            # Only plate-class boxes go through OCR
            if not is_plate_class(class_name):
                continue

            h, w = crop.shape[:2]

            if w < 50 or h < 15:
                continue

            # =================================================
            # OCR (plate crop is still RGB, since it came from
            # the original `image`)
            # =================================================

            detected_text, confidence = read_plate(
                ocr, crop, color_mode="rgb"
            )

            if detected_text:

                detected_plates.append((detected_text, confidence))

                text_y = y2 + 25

                if text_y > image.shape[0] - 10:
                    text_y = y1 - 10

                cv2.putText(
                    annotated_bgr, detected_text, (x1, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
                )

                confidence_text = f"{confidence * 100:.1f}%"

                cv2.putText(
                    annotated_bgr, confidence_text, (x1, text_y + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2
                )

    # -------------------------------------------------------
    # Convert back to RGB before handing to Gradio
    # -------------------------------------------------------

    annotated_image = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

    # -------------------------------------------------------
    # Prepare Final Result
    # -------------------------------------------------------

    if not detected_plates:
        final_text = "لم يتم التعرف على لوحة واضحة في الصورة."
    else:
        final_lines = []

        for index, (text, confidence) in enumerate(detected_plates, start=1):
            final_lines.append(
                f"اللوحة {index}: {text} (Confidence: {confidence * 100:.1f}%)"
            )

        final_text = "\n".join(final_lines)

    return annotated_image, final_text


# =========================================================
# 3. Gradio Interface
# =========================================================

interface = gr.Interface(
    fn=process_plate,
    inputs=gr.Image(
        sources=["upload", "webcam"],
        type="numpy",
        label="ارفع صورة أو استخدم الكاميرا"
    ),
    outputs=[
        gr.Image(label="الصورة بعد المعالجة"),
        gr.Textbox(label="النص المستخرج من اللوحة", lines=4)
    ],
    title="نظام التعرف على لوحات السيارات \U0001F697",
    description=(
        "قم برفع صورة لسيارة أو استخدم كاميرا الويب. "
        "سيقوم النظام باستخدام YOLO لاكتشاف لوحة السيارة "
        "ثم PaddleOCR لقراءة النص الموجود عليها."
    )
)


# =========================================================
# 4. Run Application
# =========================================================

if __name__ == "__main__":
    interface.launch(share=False, theme=gr.themes.Default())
