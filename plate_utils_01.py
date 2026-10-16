import os         # file paths and environment variables (used for the model path)
import re         # regular expressions: search/clean text by pattern (used to strip symbols from OCR text)
import cv2        # OpenCV: image processing (grayscale, resize, contrast)
import difflib    # measures how similar two strings are (used to catch misreads like JQRDAN ~ JORDAN)

# =========================================================
# Model path
# =========================================================
# By default the weights are expected at  models/best.pt  next to this file.
# Override anytime with the PLATE_MODEL_PATH environment variable, e.g.
#   PowerShell:  $env:PLATE_MODEL_PATH = "D:\weights\best.pt"
#   bash:        export PLATE_MODEL_PATH=/path/to/best.pt

MODEL_PATH = os.environ.get(
    "PLATE_MODEL_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "best.pt")
)


# =========================================================
# Text Cleaning / Validation
# =========================================================

def clean_plate_text(text):
    text = re.sub(r'[^A-Za-z0-9\u0600-\u06FF]', '', text)
    text = text.replace("ـ", "")
    return text.strip()


def is_plate_class(class_name):
    """
    True if this YOLO class represents the license plate itself
    (as opposed to a vehicle-type class like car/truck/motorcycle).
    """
    return "plate" in class_name.lower()   # lowercase the name, then check it contains "plate" (e.g. "License_Plate" -> True, "cars" -> False)


def is_valid_plate(text):
    if not text:                  # empty string (or None) -> not a plate
        return False
    if len(text) < 2:             # a single character is almost always OCR noise
        return False
    if len(text) > 12:            # too long to be a plate number
        return False
    if not re.search(r'[A-Za-z0-9\u0600-\u06FF]', text):   # must contain at least one English letter, digit, or Arabic letter
        return False
    return True                   # passed all checks


# =========================================================
# Plate Preprocessing
# =========================================================

def preprocess_plate(plate, color_mode="bgr"):
    """
    color_mode:
        "bgr" -> frame came from cv2.VideoCapture (plate_02.py)
        "rgb" -> frame came from Gradio / PIL as a numpy array (app_02.py)
    """
    if color_mode == "rgb":                                   # image came from Gradio (RGB order)
        gray = cv2.cvtColor(plate, cv2.COLOR_RGB2GRAY)        # color -> grayscale (one channel, brightness only)
    else:                                                     # image came from the camera (BGR order)
        gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)

    gray = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)   # enlarge 3x in width and height; CUBIC = smooth, high-quality resizing

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))   # CLAHE = local contrast boost; clipLimit caps how strong it gets, 8x8 = grid of small regions
    gray = clahe.apply(gray)                                      # apply it to the grayscale plate

    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)   # back to 3 channels, because PaddleOCR expects a color-shaped image


# =========================================================
# Model Loading (with clear errors instead of raw tracebacks)
# =========================================================

def load_yolo_model(model_path=None):
    from ultralytics import YOLO

    path = model_path or MODEL_PATH

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"YOLO model not found at: {path}\n"
            f"Place your weights at models/best.pt, or set the PLATE_MODEL_PATH env var."
        )

    return YOLO(path)


def load_ocr_model():
    from paddleocr import PaddleOCR   # imported here (not at the top) so the heavy library only loads when we actually need it

    return PaddleOCR(
        lang="en",                              # English recognition model (Latin letters + digits)
        use_doc_orientation_classify=False,     # skip detecting if the whole page is rotated (made for scanned documents)
        use_doc_unwarping=False,                # skip flattening curved/warped pages (not needed for a plate crop)
        use_textline_orientation=False,         # skip checking if each text line is upside down (plates are upright)
        device="cpu",                           # run on the CPU (no GPU on this machine)
        enable_mkldnn=False                     # turn off Intel acceleration, same crash-avoidance as the os.environ flags
    )


# =========================================================
# OCR Reading (shared single-plate OCR logic)
# =========================================================

# Text that shows up on the plate itself but isn't part of the plate
# number (country name/code, in Arabic or English). Matched fuzzily
# after clean_plate_text, so OCR misreads like "JQRDAN" for "JORDAN"
# still get caught.
IGNORED_PLATE_WORDS = {"jordan", "jo", "الاردن", "الأردن"}


def _looks_like_ignored_word(clean_text):
    lower = clean_text.lower()

    for word in IGNORED_PLATE_WORDS:
        if difflib.SequenceMatcher(None, lower, word).ratio() >= 0.75:
            return True

    return False


# Prints every raw OCR segment (text + confidence) before they get
# combined. Handy when tuning the filters; off by default.
OCR_DEBUG = False

# Minimum confidence for a text segment to be trusted. Raised from 0.40
# after noticing spurious low-confidence reads (e.g. "3991" at 0.61)
# getting mixed in alongside genuine plate digits at 0.99+.
MIN_OCR_CONFIDENCE = 0.75

# Max length for a single OCR segment to be treated as part of the
# plate number. A real plate line is short (2-7 chars); longer
# high-confidence reads are usually stray text printed on the plate
# frame itself (a dealer's phone number, an ad, etc.) and get dropped.
MAX_SEGMENT_LENGTH = 7


def read_plate(ocr, plate, color_mode="bgr"):
    """
    Runs OCR on a cropped plate image and returns (combined_text, avg_score).
    Returns ("", 0.0) if nothing valid was found.

    Plates that span two lines (e.g. "37" on top, "68189" below, as in
    Jordanian plates) get detected by PaddleOCR as separate text segments.
    We combine every valid segment (in the order PaddleOCR returns them,
    which follows reading order) instead of keeping only the single
    highest-confidence one — otherwise every line but the best-scoring
    one gets silently dropped. Segments that are just the country
    name/code (e.g. "JORDAN", "الأردن") are excluded from the combined
    text so they don't blow past the length limit in is_valid_plate.
    Plates can be alphanumeric (e.g. UK plates like "KY75OBH"), so
    letters are allowed through — only the country-name text is
    filtered, not letters in general.
    """
    try:
        plate_ready = preprocess_plate(plate, color_mode=color_mode)
        ocr_result = ocr.predict(plate_ready)

        segments = []

        for result in ocr_result:
            if not result:
                continue

            texts = result.get("rec_texts", [])
            scores = result.get("rec_scores", [])

            for text, score in zip(texts, scores):
                score = float(score)

                if score < MIN_OCR_CONFIDENCE:
                    continue

                clean_text = clean_plate_text(text)

                if _looks_like_ignored_word(clean_text):
                    continue

                if len(clean_text) > MAX_SEGMENT_LENGTH:
                    continue

                if not is_valid_plate(clean_text):
                    continue

                segments.append((clean_text, score))

        if not segments:
            return "", 0.0

        if OCR_DEBUG:
            print(f"[OCR-DEBUG] raw segments: {segments}")

        combined_text = "".join(s[0] for s in segments)
        avg_score = sum(s[1] for s in segments) / len(segments)

        if not is_valid_plate(combined_text):
            return "", 0.0

        return combined_text, avg_score

    except Exception as e:
        print("OCR Error:", e)
        return "", 0.0
