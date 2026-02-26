# face_verification/face_utils.py
import cv2, os, json
from datetime import datetime
BASE = os.path.join(os.path.dirname(__file__), "..")
CAP_DIR = os.path.join(BASE, "assets", "captured_faces")
os.makedirs(CAP_DIR, exist_ok=True)
HAAR = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(HAAR)

# Replace this function in face_verification/face_utils.py

def parse_bbox(bbox_input):
    """
    Accepts bbox as:
      - list or tuple: [x, y, w, h] -> returns (x,y,w,h) ints
      - JSON string: "[x, y, w, h]" -> parsed to list
      - CSV string: "x,y,w,h" -> split and parsed
    Returns tuple of ints (x, y, w, h).
    Raises ValueError if cannot parse.
    """
    import json

    # If already a list/tuple of numbers
    if isinstance(bbox_input, (list, tuple)):
        if len(bbox_input) >= 4:
            try:
                x, y, w, h = int(bbox_input[0]), int(bbox_input[1]), int(bbox_input[2]), int(bbox_input[3])
                return (x, y, w, h)
            except Exception as e:
                raise ValueError(f"Invalid bbox numeric values: {bbox_input}") from e
        else:
            raise ValueError(f"bbox list/tuple must have 4 elements, got: {bbox_input}")

    # If it's already numeric-like (unlikely)
    if isinstance(bbox_input, (int, float)):
        raise ValueError("bbox must be list/tuple or string containing 4 values")

    # At this point expect a string
    if not isinstance(bbox_input, str):
        raise ValueError(f"Unsupported bbox type: {type(bbox_input)}")

    s = bbox_input.strip()
    # Try JSON first
    try:
        obj = json.loads(s)
        if isinstance(obj, (list, tuple)) and len(obj) >= 4:
            return (int(obj[0]), int(obj[1]), int(obj[2]), int(obj[3]))
    except Exception:
        pass

    # Try comma separated numbers
    parts = [p.strip() for p in s.strip("[] ").split(",") if p.strip() != ""]
    if len(parts) >= 4:
        try:
            x, y, w, h = int(float(parts[0])), int(float(parts[1])), int(float(parts[2])), int(float(parts[3]))
            return (x, y, w, h)
        except Exception as e:
            raise ValueError(f"Could not parse bbox parts to ints: {parts}") from e

    raise ValueError(f"Unable to parse bbox input: {bbox_input}")


def crop_frame(frame, bbox):
    """
    Robust cropping helper.

    Accepts bbox as:
       - list/tuple: [x, y, w, h]  (top-left + width/height) OR
       - list/tuple: [x1, y1, x2, y2]  (two corners)
       - string JSON or CSV will already be parsed by parse_bbox() before calling this.

    Returns the cropped image (numpy array) or None when the crop is invalid/empty.
    """
    import numpy as np

    if frame is None:
        return None
    h_frame, w_frame = frame.shape[:2]

    # Expect bbox already parsed to numeric tuple/list
    if not isinstance(bbox, (list, tuple)):
        # not a sequence -> cannot crop
        return None

    if len(bbox) < 4:
        return None

    try:
        x0, y0, x1, y1 = bbox[0], bbox[1], bbox[2], bbox[3]
        # convert to ints (handle floats)
        x0 = int(round(float(x0)))
        y0 = int(round(float(y0)))
        x1 = int(round(float(x1)))
        y1 = int(round(float(y1)))
    except Exception:
        return None

    # Heuristic: detect whether bbox is (x,y,w,h) or (x1,y1,x2,y2)
    # If x1 or y1 look like width/height (i.e., small relative to frame), we treat as (x,y,w,h)
    # If x1 > frame width or y1 > frame height or x1 <= x0 or y1 <= y0, it's likely (x1,y1) corner.
    use_w_h = False
    if (x1 <= 0) or (y1 <= 0):
        use_w_h = True
    elif (x1 <= w_frame and y1 <= h_frame) and (x0 + x1 <= w_frame and y0 + y1 <= h_frame):
        # if x1 and y1 look like width/height (x0 + x1 stays inside frame), lean towards w/h
        use_w_h = True
    elif (x1 > w_frame or y1 > h_frame) or (x1 <= x0 or y1 <= y0):
        # if second values look like absolute coords but out of bounds or less than x0/y0, treat as x2/y2
        use_w_h = False
    else:
        # default: if x1 is small (< frame width*0.6) AND x0 + x1 <= frame width, treat as w/h
        if (x1 < (w_frame * 0.6) and (x0 + x1) <= w_frame) or (y1 < (h_frame * 0.6) and (y0 + y1) <= h_frame):
            use_w_h = True
        else:
            use_w_h = False

    if use_w_h:
        x = x0
        y = y0
        w = x1
        h = y1
        # if width/height possibly negative or zero, bail
        if w <= 0 or h <= 0:
            return None
        x2 = x + w
        y2 = y + h
    else:
        # treat as x1,y1,x2,y2
        x = x0
        y = y0
        x2 = x1
        y2 = y1
        # ensure x2 > x and y2 > y by swapping if needed
        if x2 < x:
            x, x2 = x2, x
        if y2 < y:
            y, y2 = y2, y
        # compute width/height
        w = x2 - x
        h = y2 - y
        if w <= 0 or h <= 0:
            return None

    # Clamp to frame bounds
    x = max(0, min(x, w_frame - 1))
    y = max(0, min(y, h_frame - 1))
    x2 = max(0, min(x2, w_frame))
    y2 = max(0, min(y2, h_frame))

    # Recompute to ensure positive size
    if x2 <= x or y2 <= y:
        return None

    try:
        crop = frame[y:y2, x:x2].copy()
        if crop.size == 0 or crop.shape[0] == 0 or crop.shape[1] == 0:
            return None
        return crop
    except Exception:
        return None


def face_present(img) -> bool:
    """Fallback Haar-based presence check (kept for robustness)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
    return len(faces) > 0


_yolo_model = None

def face_present_yolo(img) -> bool:
    """
    Use YOLOv8 (yolov8s.pt) for object detection to decide if a person/face is present.
    We treat any 'person' detection inside the crop as presence.
    """
    global _yolo_model
    try:
        from ultralytics import YOLO
    except Exception:
        # If YOLO is not available, fall back to Haar
        return face_present(img)

    if _yolo_model is None:
        model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "yolov8s.pt")
        _yolo_model = YOLO(model_path)

    try:
        results = _yolo_model.predict(source=img, verbose=False)
        if not results:
            return False
        r = results[0]
        names = r.names
        for box in r.boxes:
            cls_id = int(box.cls[0].item())
            label = names.get(cls_id, "")
            if label.lower() == "person":
                return True
        return False
    except Exception:
        # On any YOLO error, fall back to Haar
        return face_present(img)

def save_capture(img, classroom_id, seat_label, assigned_roll):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fn = f"room{classroom_id}_{seat_label}_roll{assigned_roll}_{ts}.jpg"
    path = os.path.join(CAP_DIR, fn)
    cv2.imwrite(path, img)
    return path
