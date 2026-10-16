import os

# =========================================================
# Disable Paddle features that may cause crashes
# =========================================================

os.environ["FLAGS_enable_pir_api"] = "0"            # turn off Paddle's newer PIR execution engine (unstable on some setups)
os.environ["FLAGS_use_mkldnn"] = "0"                # turn off Intel MKL-DNN acceleration (can crash on non-Intel / older CPUs)
os.environ["FLAGS_use_onednn"] = "0"                # turn off oneDNN acceleration (same reason as above)
os.environ["PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT"] = "0"  # same MKL-DNN toggle, but for the PaddleX interface

import sys                          # lets us stop the program cleanly (sys.exit) if setup fails
import cv2                          # OpenCV: camera access, image processing, drawing boxes/text
import time                         # used to measure elapsed time for the FPS counter
from collections import Counter     # counts repeated OCR readings to pick the most stable one

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


# =========================================================
# 2. Camera
# =========================================================

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not cap.isOpened():
    print("ERROR: Cannot open camera \u274c")
    sys.exit(1)

print("Starting camera... Press q to exit")


# =========================================================
# 3. Optimization Variables
# =========================================================

total_frames = 0   # counts every frame read from the camera (used for skip_frames / ocr_interval)
fps_frames = 0     # counts frames inside the current 1-second window (to compute FPS)

# Debug: print every detected box's class + confidence to the console.
# Set to False once you've confirmed what the model detects.
DEBUG = False   # True = print every detected box (class + confidence) in the terminal

# Lowered from 0.5 so weaker vehicle-type detections aren't silently
# dropped before you can even see them in the debug print below.
yolo_conf = 0.25   # YOLO drops any box with confidence below this (lower = more boxes, more noise)

# Skip YOLO processing on some frames.
# Raised from 2 -> 3 to reduce load on weaker CPUs.
skip_frames = 3   # run YOLO on 1 out of every 3 frames (the others are just displayed)

# OCR is the heaviest step, so it only runs when total_frames is a multiple of ocr_interval.
# NOTE: YOLO already only runs when total_frames is a multiple of skip_frames (3), so the
# two conditions only line up when total_frames is a multiple of both = every 30 frames.
ocr_interval = 10   # counts raw camera frames (not processed ones); effectively OCR runs every 30 frames

fps = 0                  # the FPS value shown on screen
fps_time = time.time()   # timestamp when the current 1-second window started


# =========================================================
# 4. OCR Stabilization
# =========================================================

ocr_history = []            # list of the most recent OCR readings (newest at the end)
max_history = 5             # keep only the last 5 readings, older ones are dropped
last_ocr_text = ""          # the stable plate text currently drawn on screen
last_printed_plate = ""     # the last plate printed in the terminal (to avoid printing the same one repeatedly)


# =========================================================
# 5. Detection Loop
# =========================================================

while True:   # main loop: runs forever, once per camera frame, until we break out of it

    success, frame = cap.read()   # grab one frame; success=False if the camera failed

    if not success:   # camera stopped giving frames -> stop the loop
        print("ERROR: Failed to read camera frame")
        break

    total_frames += 1   # one more frame seen overall
    fps_frames += 1     # one more frame in the current 1-second window

    # -----------------------------------------------------
    # FPS Calculation
    # -----------------------------------------------------

    current_time = time.time()   # current clock time, in seconds

    if current_time - fps_time >= 1:   # has a full second passed since the window started?
        fps = fps_frames               # frames counted in that second = FPS
        fps_frames = 0                 # reset the counter for the next second
        fps_time = current_time        # start a new 1-second window

    # -----------------------------------------------------
    # Skip Frames
    # -----------------------------------------------------

    if total_frames % skip_frames != 0:   # not every 3rd frame -> skip YOLO, just show the raw frame

        cv2.putText(                          # write text directly on the image
            frame, f"FPS: {fps}", (10, 30),   # image, the text, position (x=10, y=30) = top-left corner
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2   # font, size, color (green, BGR), thickness
        )

        cv2.imshow("YOLO License Plate OCR", frame)   # show the frame in a window with this title

        if cv2.waitKey(1) & 0xFF == ord('q'):   # wait 1ms for a key; 'q' = quit
            break

        continue   # jump straight to the next frame (skips everything below)

    annotated_frame = frame.copy()   # copy to draw on, so the original frame stays clean for cropping the plate

    # -----------------------------------------------------
    # YOLO Detection
    # -----------------------------------------------------

    results = model(frame, imgsz=640, conf=yolo_conf, verbose=False)   # run YOLO on the frame: imgsz = internal resize, conf = min confidence, verbose=False = no console spam

    for r in results:

        if r.boxes is None:   # nothing detected in this result -> skip it
            continue

        boxes = r.boxes.xyxy.cpu().numpy()       # box corners [x1, y1, x2, y2] as a numpy array
        class_ids = r.boxes.cls.cpu().numpy()    # class number of each box (0, 1, 2, 3)
        confs = r.boxes.conf.cpu().numpy()       # how confident YOLO is about each box (0 to 1)

        for box, cls_id, conf in zip(boxes, class_ids, confs):   # go through the detected boxes one by one

            x1, y1, x2, y2 = map(int, box)   # top-left (x1,y1) and bottom-right (x2,y2), converted to whole pixels

            x1 = max(0, x1)                    # keep the box inside the image (no negative coordinates)
            y1 = max(0, y1)
            x2 = min(frame.shape[1], x2)       # frame.shape[1] = image width
            y2 = min(frame.shape[0], y2)       # frame.shape[0] = image height

            class_name = model.names[int(cls_id)]   # turn the class number into its name (e.g. 0 -> License_Plate)

            if DEBUG:
                print(f"[DEBUG] class={class_name}  conf={conf:.2f}")

            crop = frame[y1:y2, x1:x2]   # cut this box out of the image (rows y1..y2, columns x1..x2)

            if crop.size == 0:           # empty crop (bad box) -> skip
                continue

            # ===============================================
            # Draw YOLO Bounding Box + Class Label
            # (works for both vehicle-type and plate boxes)
            # ===============================================

            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)   # draw the box (blue in BGR, thickness 2)

            label_y = y1 - 10        # put the label just above the box

            if label_y < 20:         # no room above (box touches the top edge)
                label_y = y1 + 20    # -> put the label inside the box instead

            cv2.putText(
                annotated_frame, class_name, (x1, label_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2   # yellow label text
            )

            # Only plate-class boxes go through OCR
            if not is_plate_class(class_name):   # car / truck / motorcycle -> label only, no OCR
                continue

            h, w = crop.shape[:2]   # height and width of the cropped plate

            if w < 50 or h < 15:    # too small to read -> skip
                continue

            # ===============================================
            # OCR (only when total_frames is a multiple of ocr_interval,
            # which in practice means every 30 frames because of skip_frames)
            # ===============================================

            if total_frames % ocr_interval == 0:   # raw-frame counter: true on frames 10, 20, 30... (but YOLO only reaches here on 3, 6, 9... so effectively 30, 60, 90...)

                detected_text, best_score = read_plate(   # read the plate text -> (text, confidence)
                    ocr, crop, color_mode="bgr"
                )

                if detected_text:   # OCR actually returned something

                    ocr_history.append(detected_text)   # remember this reading

                    if len(ocr_history) > max_history:  # more than 5 readings stored?
                        ocr_history.pop(0)              # -> drop the oldest one

                    counter = Counter(ocr_history)   # count how many times each reading appears
                    stable_text, stable_count = counter.most_common(1)[0]   # the most repeated reading and its count

                    if stable_count >= 2 or len(ocr_history) == 1:   # repeated at least twice (or it's the very first reading)

                        last_ocr_text = stable_text                  # -> accept it as the plate text to display

                        if stable_text != last_printed_plate:   # print only when the plate changes

                            print(
                                "Detected Plate:", stable_text,
                                "| Confidence:", round(best_score * 100, 2), "%"
                            )

                            last_printed_plate = stable_text   # remember what we just printed

            # ===============================================
            # Draw OCR Text (below the box, since the class
            # label already sits above it)
            # ===============================================

            if last_ocr_text:   # if we have a stable plate text, draw it

                text_y = y2 + 25   # position: just below the box

                if text_y > frame.shape[0] - 10:   # no room below (box near the bottom edge)
                    text_y = y1 - 10               # -> draw above the box instead

                cv2.putText(
                    annotated_frame, last_ocr_text, (x1, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2   # green plate text
                )

    # -----------------------------------------------------
    # Draw FPS + Show Frame
    # -----------------------------------------------------

    cv2.putText(                                  # draw the FPS counter on the annotated frame
        annotated_frame, f"FPS: {fps}", (10, 30),   # top-left corner
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2   # green text
    )

    cv2.imshow("YOLO License Plate OCR", annotated_frame)   # show the final frame (boxes + labels + plate text + FPS)

    if cv2.waitKey(1) & 0xFF == ord('q'):   # wait 1ms for a key press; if it is 'q'...
        break                                  # ...leave the main loop


# =========================================================
# Cleanup
# =========================================================

cap.release()               # free the camera so other programs can use it
cv2.destroyAllWindows()     # close the OpenCV window(s)

print("Program stopped successfully.")   # confirm in the terminal that we exited cleanly
