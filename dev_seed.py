# dev_seed.py — quick helper to create test seats and one student for classroom 2
import json
from database import db_utils
import os

CLASSROOM_ID = 2   # change if your classroom id is different

# 1) Create two seats with simple bboxes (x,y,w,h)
# Choose small boxes in the upper-left of the frame for quick testing.
seats = [
    {"seat_label": "Seat-1", "bbox": json.dumps([50, 50, 200, 200]), "row": 1, "col": 1, "assigned_roll_no": 101},
    {"seat_label": "Seat-2", "bbox": json.dumps([300, 50, 200, 200]), "row": 1, "col": 2, "assigned_roll_no": 102},
]

for s in seats:
    sid = db_utils.insert_or_update_seat(CLASSROOM_ID, s["seat_label"], s["bbox"], s["row"], s["col"], s["assigned_roll_no"])
    print("Upserted seat id:", sid, "label:", s["seat_label"])

# 2) Add a sample student record (photo must exist or upload via UI)
# Place a test image at assets/student_photos/roll_101.jpg (or change path below)
photo_path = os.path.join("assets", "student_photos", "roll_101.jpg")
# If the file doesn't exist, create a tiny placeholder image so DeepFace won't find a face but DB test still works.
if not os.path.exists(photo_path):
    from PIL import Image
    os.makedirs(os.path.dirname(photo_path), exist_ok=True)
    Image.new("RGB", (200,200), (200,200,200)).save(photo_path)
    print("Created placeholder student image at", photo_path)

db_utils.insert_student(101, "Test Student 101", photo_path)
print("Inserted student 101 ->", photo_path)
