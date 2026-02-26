# run_verification.py
import cv2
import os, time
from database.db_utils import get_seats_by_classroom, get_student, insert_verification_log, update_seat_status
from face_verification.face_utils import parse_bbox, crop_frame, face_present_yolo, save_capture
from face_verification.verify_faces import verify

# Default camera fallback
DEFAULT_CAMERA = 0

def run_once(classroom_id:int, camera_source=None):
    """
    Single-pass verification for classroom_id using camera_source.
    camera_source may be int (0) or RTSP string. This function is safe to call from a thread.
    """
    if camera_source is None:
        camera_source = DEFAULT_CAMERA

    # initialize capture
    try:
        cap = cv2.VideoCapture(int(camera_source)) if isinstance(camera_source, (int, str)) and str(camera_source).isdigit() else cv2.VideoCapture(camera_source)
    except Exception:
        cap = cv2.VideoCapture(camera_source)

    if not cap.isOpened():
        print(f"[run_once] ERROR: cannot open camera {camera_source}")
        return

    # warm-up
    time.sleep(0.8)
    ret, frame = cap.read()
    if not ret or frame is None:
        print("[run_once] ERROR: unable to read frame from camera")
        cap.release()
        return

    seats = get_seats_by_classroom(classroom_id)
    if not seats:
        print("[run_once] No seats configured for classroom", classroom_id)
        cap.release()
        return

    for seat in seats:
        seat_id = seat['id']; seat_label = seat.get('seat_label') or f"seat_{seat_id}"
        bbox_text = seat.get('bbox')
        assigned_roll = seat.get('assigned_roll_no')

        if not bbox_text:
            insert_verification_log(classroom_id, seat_id, seat_label, assigned_roll, None, "error", "no_bbox", None, None)
            update_seat_status(seat_id, "no_bbox")
            continue

        bbox = parse_bbox(bbox_text)
        crop = crop_frame(frame, bbox)
        if crop is None or crop.size == 0:
            insert_verification_log(classroom_id, seat_id, seat_label, assigned_roll, None, "error", "empty_crop", None, None)
            update_seat_status(seat_id, "error")
            continue

        # Use YOLO-based presence check (falls back to Haar if YOLO unavailable)
        if not face_present_yolo(crop):
            saved = save_capture(crop, classroom_id, seat_label, assigned_roll or 0)
            insert_verification_log(classroom_id, seat_id, seat_label, assigned_roll, None, "absent", "no_face_detected_yolo", None, saved)
            update_seat_status(seat_id, "absent")
            continue

        if not assigned_roll:
            saved = save_capture(crop, classroom_id, seat_label, 0)
            insert_verification_log(classroom_id, seat_id, seat_label, None, None, "unassigned", "no_assigned_roll", None, saved)
            update_seat_status(seat_id, "unassigned")
            continue

        student = get_student(assigned_roll)
        if not student:
            saved = save_capture(crop, classroom_id, seat_label, assigned_roll)
            insert_verification_log(classroom_id, seat_id, seat_label, assigned_roll, None, "missing_photo", "student_not_in_db", None, saved)
            update_seat_status(seat_id, "missing_photo")
            continue

        captured_path = save_capture(crop, classroom_id, seat_label, assigned_roll)
        student_photo = student.get('photo_path')
        if not student_photo or not os.path.exists(student_photo):
            insert_verification_log(classroom_id, seat_id, seat_label, assigned_roll, None, "missing_photo_file", "student_photo_missing_file", None, captured_path)
            update_seat_status(seat_id, "missing_photo_file")
            continue

        # run deepface verify (may take time)
        res = verify(student_photo, captured_path, enforce_detection=False)
        verified = res.get('verified', False)
        distance = res.get('distance', None)

        if verified:
            insert_verification_log(classroom_id, seat_id, seat_label, assigned_roll, assigned_roll, "verified", "match", distance, captured_path)
            update_seat_status(seat_id, "verified")
        else:
            insert_verification_log(classroom_id, seat_id, seat_label, assigned_roll, None, "mismatch", "not_match", distance, captured_path)
            update_seat_status(seat_id, "mismatch")

    cap.release()
    print(f"[run_once] verification pass for classroom {classroom_id} finished.")
