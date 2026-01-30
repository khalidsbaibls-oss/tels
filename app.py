from fastapi import FastAPI, File, UploadFile
from ultralytics import YOLO
from PIL import Image
import io
import re
import statistics

# ==========================
# CONFIG
# ==========================
MODEL_PATH = "best (14).pt"
CONF_THRESHOLD = 0.25

MIN_BOX_AREA_NORM = 0.0022
MIN_BOX_HEIGHT_NORM = 0.035

app = FastAPI(title="YOLO Phone Number API")

# ==========================
# LOAD MODEL ONCE
# ==========================
model = YOLO(MODEL_PATH)

# ==========================
# HELPERS
# ==========================
def compute_auto_y_threshold(detections):
    if not detections:
        return 0.07
    heights = [d["h"] for d in detections]
    return statistics.median(heights) * 1.5


def clean_number(s):
    return re.sub(r"\D", "", s)

# ==========================
# API ENDPOINT
# ==========================
@app.post("/detect")
async def detect_phone_numbers(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    results = model.predict(image, conf=CONF_THRESHOLD, verbose=False)

    phone_numbers = []

    for r in results:
        detections = []

        for box in r.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            if conf < CONF_THRESHOLD:
                continue

            x, y, w, h = box.xywhn[0].tolist()

            if (w * h) < MIN_BOX_AREA_NORM:
                continue
            if h < MIN_BOX_HEIGHT_NORM:
                continue

            detections.append({
                "x": x,
                "y": y,
                "h": h,
                "cls": cls
            })

        if not detections:
            continue

        Y_THRESHOLD = compute_auto_y_threshold(detections)

        # ---- Group by Y proximity ----
        lines = []
        for d in detections:
            placed = False
            for line in lines:
                avg_y = sum(i["y"] for i in line) / len(line)
                if abs(avg_y - d["y"]) < Y_THRESHOLD:
                    line.append(d)
                    placed = True
                    break
            if not placed:
                lines.append([d])

        # ---- Build numbers ----
        for line in lines:
            line.sort(key=lambda d: d["x"])
            number = clean_number("".join(str(d["cls"]) for d in line))
            if number:
                phone_numbers.append(number)

    return {
        "count": len(phone_numbers),
        "numbers": phone_numbers
    }
