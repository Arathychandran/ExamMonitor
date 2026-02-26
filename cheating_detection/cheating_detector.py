import cv2
import time
from collections import deque
from typing import List, Dict, Any, Optional, Tuple

from database.db_utils import get_seats_by_classroom
from face_verification.face_utils import parse_bbox, crop_frame

# Optional: MediaPipe for richer head/hand/mouth indicators.
# We treat it as strictly best-effort; if anything about the install
# looks wrong (e.g. no "solutions" attribute), we just fall back to
# the motion-based heuristics instead of crashing.
_MP_AVAILABLE = False
mp_face_mesh_mod = None
mp_hands_mod = None
try:
    import mediapipe as mp  # type: ignore
    if hasattr(mp, "solutions"):
        mp_face_mesh_mod = mp.solutions.face_mesh
        mp_hands_mod = mp.solutions.hands
        _MP_AVAILABLE = True
    else:
        _MP_AVAILABLE = False
except Exception:
    _MP_AVAILABLE = False

# Optional: YOLO object detector (for phone detection). Best-effort.
_YOLO_AVAILABLE = False
try:
    from ultralytics import YOLO  # type: ignore
    _YOLO_AVAILABLE = True
except Exception:
    _YOLO_AVAILABLE = False


def analyze_classroom(
    classroom_id: int,
    camera_source: Optional[str] = None,
    duration_seconds: int = 10,
    target_fps: int = 5,
) -> List[Dict[str, Any]]:
    """
    Lightweight, fully offline cheating-analysis pass.

    Assumptions:
      - Seat bounding boxes are fixed and one student per box.
      - We do NOT run person detection or tracking; we only analyze motion
        inside each predefined box.

    Heuristic (first version):
      - For each seat, we maintain a sliding window of frame-difference scores
        between consecutive crops.
      - Excessive, frequent motion over the window is flagged as Suspicious
        (cheating prediction: "excessive_movement").

    Returns a list of dicts:
      {
        "seat_label": str,
        "assigned_roll_no": int or None,
        "status": "Normal" | "Suspicious",
        "cheating_type": str,
        "confidence": float (0-100)
      }
    """
    if camera_source is None:
        camera_source = 0

    cap = None
    try:
        cap = cv2.VideoCapture(
            int(camera_source)
        ) if isinstance(camera_source, (int, str)) and str(camera_source).isdigit() else cv2.VideoCapture(
            camera_source
        )
    except Exception:
        cap = cv2.VideoCapture(camera_source)

    if not cap or not cap.isOpened():
        print(f"[cheating] cannot open camera {camera_source}")
        return []

    seats = get_seats_by_classroom(classroom_id)
    if not seats:
        print(f"[cheating] no seats configured for classroom {classroom_id}")
        cap.release()
        return []

    # Per-seat history:
    #  - motion scores (frame differences)
    #  - centroid shifts (dx, dy) to approximate head turns / posture changes
    history: Dict[int, deque] = {}
    dir_history: Dict[int, deque] = {}
    # Per-seat indicator counters over the whole analysis window
    indicator_counts: Dict[int, Dict[str, float]] = {}
    last_crops_gray: Dict[int, Any] = {}
    last_centroids: Dict[int, Tuple[float, float]] = {}
    last_mouth_regions: Dict[int, Any] = {}  # For fallback mouth detection
    last_hand_centroids: Dict[int, Tuple[float, float]] = {}  # For hand-motion detection
    window_size = max(5, duration_seconds * target_fps // 2)

    start = time.time()
    frame_interval = 1.0 / max(1, target_fps)

    # Optional MediaPipe models
    face_mesh = None
    hands_detector = None
    if _MP_AVAILABLE and mp_face_mesh_mod is not None and mp_hands_mod is not None:
        face_mesh = mp_face_mesh_mod.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        hands_detector = mp_hands_mod.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    # Optional YOLO model for phone detection (lazy init)
    yolo_model = None
    yolo_names = None
    if _YOLO_AVAILABLE:
        try:
            model_path = cv2.os.path.join(cv2.os.path.dirname(cv2.os.path.dirname(__file__)), "yolov8s.pt")
            # cv2.os is not real on some builds; fallback to os if needed
        except Exception:
            model_path = None
        try:
            import os as _os
            if model_path is None:
                model_path = _os.path.join(_os.path.dirname(_os.path.dirname(__file__)), "yolov8s.pt")
            yolo_model = YOLO(model_path)
            # names available on results[0].names; keep None here
        except Exception:
            yolo_model = None

    while time.time() - start < duration_seconds:
        ok, frame = cap.read()
        if not ok or frame is None:
            time.sleep(0.05)
            continue

        frame_h, frame_w = frame.shape[:2]

        for seat in seats:
            seat_id = seat["id"]
            bbox_raw = seat.get("bbox")
            if not bbox_raw:
                continue
            try:
                bbox = parse_bbox(bbox_raw)
                crop = crop_frame(frame, bbox)
                if crop is None or crop.size == 0:
                    continue
            except Exception:
                continue

            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (64, 64))

            if seat_id not in history:
                history[seat_id] = deque(maxlen=window_size)
                dir_history[seat_id] = deque(maxlen=window_size)
                indicator_counts[seat_id] = {
                    "head_turns": 0.0,
                    "gaze_away": 0.0,
                    "hand_events": 0.0,
                    "hand_motion_events": 0.0,
                    "phone_events": 0.0,
                    "mouth_events": 0.0,
                    "frames": 0.0,
                }

            # centroid of intensity (very rough "head" location proxy)
            cy, cx = cv2.moments(gray)["m01"], cv2.moments(gray)["m10"]
            m00 = cv2.moments(gray)["m00"]
            if m00 != 0:
                cx /= m00
                cy /= m00
            else:
                cx, cy = gray.shape[1] / 2.0, gray.shape[0] / 2.0

            if seat_id in last_crops_gray:
                diff = cv2.absdiff(gray, last_crops_gray[seat_id])
                score = float(diff.mean())
                history[seat_id].append(score)

                # direction of movement between centroids
                last_cx, last_cy = last_centroids.get(seat_id, (cx, cy))
                dx = cx - last_cx
                dy = cy - last_cy
                dir_label = "still"
                mag = max(abs(dx), abs(dy))
                # More sensitive threshold to detect head movements
                # Use separate thresholds for horizontal vs vertical to better detect head tilting
                dir_thresh_horizontal = 1.0  # pixels for left/right
                dir_thresh_vertical = 0.8    # pixels for up/down (more sensitive)
                if mag > min(dir_thresh_horizontal, dir_thresh_vertical):
                    if abs(dx) >= abs(dy):
                        # Horizontal movement
                        if abs(dx) > dir_thresh_horizontal:
                            dir_label = "left" if dx < 0 else "right"
                    else:
                        # Vertical movement (head tilting up/down)
                        if abs(dy) > dir_thresh_vertical:
                            dir_label = "up" if dy < 0 else "down"
                dir_history[seat_id].append(dir_label)
            else:
                # seed with zero motion
                history[seat_id].append(0.0)
                dir_history[seat_id].append("still")

            # --- MediaPipe-based indicators (if available) ---
            ind = indicator_counts[seat_id]
            ind["frames"] += 1.0
            
            # Fallback: Use OpenCV Haar cascade for mouth detection if MediaPipe fails
            mouth_detected_this_frame = False
            hand_detected_this_frame = False
            
            if face_mesh is not None or hands_detector is not None:
                try:
                    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                    # Head pose / gaze / mouth via face mesh
                    if face_mesh is not None:
                        fm_res = face_mesh.process(rgb)
                        if fm_res.multi_face_landmarks:
                            fl = fm_res.multi_face_landmarks[0]
                            h, w = crop.shape[:2]
                            # use simple points: nose tip and eye corners
                            # indices based on MediaPipe FaceMesh topology
                            nose = fl.landmark[1]
                            left_eye = fl.landmark[33]
                            right_eye = fl.landmark[263]
                            # horizontal angle approx: difference in eye y divided by distance
                            eye_vec_x = (right_eye.x - left_eye.x)
                            eye_vec_y = (right_eye.y - left_eye.y)
                            yaw_approx = abs(eye_vec_y) / max(1e-5, abs(eye_vec_x))
                            head_turn_thresh = 0.25  # tuned empirically
                            if yaw_approx > head_turn_thresh:
                                ind["head_turns"] += 1.0

                            # mouth open ratio -> possible speaking
                            # Use more reliable landmarks for mouth detection
                            # MediaPipe FaceMesh landmarks: 13=upper lip, 14=lower lip (alternative: 12, 15 for corners)
                            upper_lip = fl.landmark[13]
                            lower_lip = fl.landmark[14]
                            mouth_open = abs(upper_lip.y - lower_lip.y) * h
                            # Lower threshold to detect more subtle mouth movements (speaking, signaling)
                            mouth_thresh = 2.5  # pixels (lowered from 4.0 for better sensitivity)
                            if mouth_open > mouth_thresh:
                                ind["mouth_events"] += 1.0
                                mouth_detected_this_frame = True

                            # very rough gaze-away: nose far from crop center
                            nose_px_x = nose.x * w
                            nose_px_y = nose.y * h
                            centre_x, centre_y = w / 2.0, h / 2.0
                            dist_centre = ((nose_px_x - centre_x) ** 2 + (nose_px_y - centre_y) ** 2) ** 0.5
                            if dist_centre > 0.35 * max(w, h):
                                ind["gaze_away"] += 1.0

                    # Hand activity near desk area
                    if hands_detector is not None:
                        hd_res = hands_detector.process(rgb)
                        if hd_res.multi_hand_landmarks:
                            ind["hand_events"] += 1.0
                            hand_detected_this_frame = True
                            # Estimate hand centroid (average of wrist points for stability)
                            # Landmark 0 is wrist
                            try:
                                wrists = []
                                for hl in hd_res.multi_hand_landmarks:
                                    wlm = hl.landmark[0]
                                    wrists.append((wlm.x, wlm.y))
                                if wrists:
                                    hx = sum(p[0] for p in wrists) / len(wrists)
                                    hy = sum(p[1] for p in wrists) / len(wrists)
                                    # Measure motion between frames (normalized coords)
                                    if seat_id in last_hand_centroids:
                                        px, py = last_hand_centroids[seat_id]
                                        if abs(hx - px) + abs(hy - py) > 0.03:
                                            ind["hand_motion_events"] += 1.0
                                    last_hand_centroids[seat_id] = (hx, hy)
                            except Exception:
                                pass
                except Exception as e:
                    # Debug: print MediaPipe errors (only once per seat)
                    pass
            
            # Fallback: Use motion-based mouth detection if MediaPipe didn't detect
            # Track motion in the lower third of the crop (mouth region)
            if not mouth_detected_this_frame:
                try:
                    h_crop, w_crop = crop.shape[:2]
                    # Focus on lower third of face (mouth region)
                    mouth_region = crop[int(h_crop * 0.6):h_crop, :]
                    if mouth_region.size > 0:
                        # Convert to grayscale and calculate motion
                        mouth_gray = cv2.cvtColor(mouth_region, cv2.COLOR_BGR2GRAY)
                        # Store previous mouth region for comparison
                        if seat_id in last_mouth_regions:
                            prev_mouth = last_mouth_regions[seat_id]
                            diff = cv2.absdiff(mouth_gray, prev_mouth)
                            motion_score = diff.mean()
                            # If there's significant motion in mouth region, likely speaking/mouth movement
                            # Lower threshold to detect subtle lip movements
                            mouth_motion_thresh = 3.0  # threshold for mouth region motion (lowered from 8.0)
                            if motion_score > mouth_motion_thresh:
                                ind["mouth_events"] += 1.0
                        last_mouth_regions[seat_id] = mouth_gray
                except Exception:
                    pass
            
            # Additional fallback: Use overall motion in crop as indicator of activity (including lip movement)
            # If student is sitting still but moving lips, there will still be some motion
            if seat_id in last_crops_gray and not mouth_detected_this_frame:
                try:
                    # Check if there's any motion at all - could be lip movement
                    current_motion = scores[-1] if scores else 0.0
                    # Lower threshold to catch subtle lip movements
                    if current_motion > 3.0:  # Any motion above noise level
                        # Increment mouth events more frequently for better detection
                        if len(scores) % 2 == 0:  # Every 2nd frame with motion
                            ind["mouth_events"] += 0.8  # Higher weight for this fallback
                except Exception:
                    pass

            # Phone detection (object detection) inside the seat crop.
            # To keep it lighter, run only when a hand is detected AND only every ~3 frames.
            if yolo_model is not None and hand_detected_this_frame:
                try:
                    # throttle: run on 1/3 frames per seat
                    if int(ind["frames"]) % 3 == 0:
                        res = yolo_model.predict(source=crop, verbose=False)
                        if res:
                            r0 = res[0]
                            names = getattr(r0, "names", None) or {}
                            # 'cell phone' is a COCO class for yolov8s
                            phone_found = False
                            for b in getattr(r0, "boxes", []):
                                cls_id = int(b.cls[0].item())
                                label = str(names.get(cls_id, "")).lower()
                                if label in ("cell phone", "mobile phone", "phone"):
                                    phone_found = True
                                    break
                            if phone_found:
                                ind["phone_events"] += 1.0
                except Exception:
                    pass

            last_crops_gray[seat_id] = gray
            last_centroids[seat_id] = (cx, cy)

        # throttle FPS
        time.sleep(frame_interval)

    cap.release()
    if face_mesh is not None:
        face_mesh.close()
    if hands_detector is not None:
        hands_detector.close()

    # Summarise per-seat behaviour
    results: List[Dict[str, Any]] = []
    base_threshold = 10.0  # motion score threshold, tuned for 0-255 grayscale

    for seat in seats:
        seat_id = seat["id"]
        label = seat.get("seat_label") or f"Seat-{seat_id}"
        assigned_roll = seat.get("assigned_roll_no")
        scores = list(history.get(seat_id, []))
        dirs = list(dir_history.get(seat_id, []))
        if not scores:
            results.append(
                {
                    "seat_label": label,
                    "assigned_roll_no": assigned_roll,
                    "status": "Normal",
                    "cheating_type": "none",
                    "reason": "insufficient_data",
                    "confidence": 0.0,
                }
            )
            continue

        avg_motion = sum(scores) / len(scores)

        # ---------- HEAD MOVEMENT INDICATOR ----------
        # Detect repeated vertical head movement (up/down) - head tilting/nodding
        # We look for:
        # 1. Consecutive streaks of up/down movements (4-5+ frames)
        # 2. Total count of up/down movements (as backup indicator)
        head_move_streak = 0
        max_head_move_streak = 0
        prev_dir = None
        up_down_count = 0  # total count of up/down movements
        
        for d in dirs:
            if d in ("up", "down"):
                up_down_count += 1
                # If previous direction was also up/down, continue streak
                if prev_dir in ("up", "down"):
                    head_move_streak += 1
                else:
                    # Start new streak
                    head_move_streak = 1
                prev_dir = d
                # Update max streak
                if head_move_streak > max_head_move_streak:
                    max_head_move_streak = head_move_streak
            else:
                # Reset streak when we see non-up/down movement
                head_move_streak = 0
                prev_dir = d

        # Threshold: if we see 2 or more consecutive up/down movements,
        # OR if total up/down movements exceed 12% of frames with at least 3 movements, flag as suspicious
        head_movement_threshold = 2  # Lowered to detect smaller streaks
        head_movement_total_threshold_ratio = 0.12  # 12% of total frames (lowered)
        head_movement_min_count = 3  # Minimum total movements required (lowered from 5)

        # ---------- LIP MOVEMENT INDICATOR ----------
        # Track sustained lip movement (mouth opening/closing) over time
        # MediaPipe-based counts (if available)
        inds = indicator_counts.get(seat_id, {"head_turns": 0.0, "gaze_away": 0.0, "hand_events": 0.0, "mouth_events": 0.0, "frames": 0.0})
        frames_total = max(1.0, inds.get("frames", 1.0))
        mp_mouth_events = int(inds.get("mouth_events", 0.0))
        
        # Calculate lip movement duration in seconds
        # Flag as suspicious if lip movement detected for more than 2 seconds (lowered for better detection)
        lip_movement_threshold_seconds = 2.0  # 2 seconds threshold (lowered from 3.0)
        lip_movement_detected = False
        lip_movement_frames = mp_mouth_events
        duration_seconds = 0.0
        lip_movement_ratio = 0.0
        
        # Calculate duration: frames with mouth open * time per frame
        if lip_movement_frames > 0:
            duration_seconds = lip_movement_frames * frame_interval
            # Flag if duration exceeds threshold (2 seconds)
            if duration_seconds >= lip_movement_threshold_seconds:
                lip_movement_detected = True
        # Also check ratio-based detection as backup (if 15%+ of frames show mouth movement)
        elif frames_total >= 10:  # Need at least 10 frames analyzed
            lip_movement_ratio = lip_movement_frames / frames_total
            if lip_movement_ratio >= 0.15:  # 15% of frames
                lip_movement_detected = True
                duration_seconds = lip_movement_frames * frame_interval
        
        # Also calculate ratio for debug purposes
        if frames_total > 0:
            lip_movement_ratio = lip_movement_frames / frames_total
        
        # Debug: print what we detected (after variables are defined)
        total_frames_debug = len(dirs)
        up_down_ratio_debug = up_down_count / max(1, total_frames_debug)
        mp_faces_detected = inds.get("frames", 0.0) > 0
        print(f"[cheating] Seat {label}: total_frames={total_frames_debug}, up/down={up_down_count} ({up_down_ratio_debug*100:.1f}%), max_streak={max_head_move_streak}, lip_movement={lip_movement_frames} frames ({duration_seconds:.2f}s, threshold {lip_movement_threshold_seconds:.0f}s), mp_faces={mp_faces_detected}, dirs_sample={dirs[:15] if len(dirs) >= 15 else dirs}")

        # Determine status based on indicators
        # Show BOTH indicators if both are detected
        status = "Normal"
        cheating_type = "none"
        reason = "no_repeated_head_movement"
        confidence = max(0.0, min(30.0, (avg_motion / base_threshold) * 30.0))

        # Head movement detection
        # Check for consecutive streak first, then total count as backup
        head_movement_detected = False
        head_movement_reason = ""
        total_frames = len(dirs)
        up_down_ratio = up_down_count / max(1, total_frames)
        
        if max_head_move_streak >= head_movement_threshold:
            head_movement_detected = True
            head_movement_reason = (
                f"repeated head movement (up/down) "
                f"{max_head_move_streak} consecutive frames "
                f"(threshold {head_movement_threshold})"
            )
        elif up_down_ratio >= head_movement_total_threshold_ratio and up_down_count >= head_movement_min_count:
            # Alternative: if total up/down movements exceed threshold ratio with minimum count
            head_movement_detected = True
            head_movement_reason = (
                f"repeated head movement (up/down) "
                f"{up_down_count} total movements ({up_down_ratio*100:.1f}% of frames, threshold {head_movement_total_threshold_ratio*100:.0f}%)"
            )
        
        # Lip movement detection
        lip_movement_reason = ""
        if lip_movement_detected:
            lip_movement_reason = (
                f"lip movement detected for {duration_seconds:.1f} seconds "
                f"(threshold {lip_movement_threshold_seconds:.0f} seconds)"
            )

        # ---------- HAND ACTIVITY INDICATOR ----------
        # Frequent hand presence/motion over the analysis window; optionally phone usage.
        hand_detected = False
        hand_reason = ""
        phone_reason = ""
        inds2 = indicator_counts.get(seat_id, {})
        frames_total2 = max(1.0, float(inds2.get("frames", 0.0) or 1.0))
        hand_frames = float(inds2.get("hand_events", 0.0) or 0.0)
        hand_motion_frames = float(inds2.get("hand_motion_events", 0.0) or 0.0)
        phone_frames = float(inds2.get("phone_events", 0.0) or 0.0)

        # Threshold: if hands are detected in >= 25% of frames OR hand-motion events >= 15% of frames.
        hand_events_threshold = max(5, int(0.25 * frames_total2))
        hand_motion_threshold = max(5, int(0.15 * frames_total2))
        if int(hand_frames) >= hand_events_threshold or int(hand_motion_frames) >= hand_motion_threshold:
            hand_detected = True
            hand_reason = (
                f"frequent hand movements detected in {int(hand_frames)} frames "
                f"(threshold {hand_events_threshold})"
            )

        # Phone threshold (very low): if detected at least once while hands present.
        if int(phone_frames) >= 1:
            phone_reason = f"phone detected in {int(phone_frames)} frames"
        
        # Combine indicators if multiple detected; show all reasons.
        triggered = []
        if head_movement_detected:
            triggered.append(("head", head_movement_reason))
        if lip_movement_detected:
            triggered.append(("lip", lip_movement_reason))
        if hand_detected:
            triggered.append(("hand", hand_reason))
        if phone_reason:
            triggered.append(("phone", phone_reason))

        if len(triggered) >= 2:
            status = "Suspicious"
            cheating_type = "_and_".join([t[0] for t in triggered])
            reason = "; ".join([t[1] for t in triggered if t[1]])
            confidence = min(100.0, max(confidence, 80.0))
        elif head_movement_detected:
            status = "Suspicious"
            cheating_type = "repeated_head_movement"
            reason = head_movement_reason
            # Confidence scaled with streak length
            if max_head_move_streak >= head_movement_threshold:
                confidence = min(100.0, 60.0 + (max_head_move_streak - head_movement_threshold) * 8.0)
            else:
                confidence = min(100.0, 55.0 + (up_down_ratio - head_movement_total_threshold_ratio) * 200.0)
        elif lip_movement_detected:
            status = "Suspicious"
            cheating_type = "lip_movement"
            reason = lip_movement_reason
            # Confidence based on duration
            confidence = min(100.0, 65.0 + (duration_seconds - lip_movement_threshold_seconds) * 10.0)
        elif hand_detected:
            status = "Suspicious"
            cheating_type = "hand_activity"
            reason = hand_reason + (f"; {phone_reason}" if phone_reason else "")
            confidence = min(100.0, 65.0 + (hand_frames / frames_total2) * 50.0)

        results.append(
            {
                "seat_label": label,
                "assigned_roll_no": assigned_roll,
                "status": status,
                    "cheating_type": cheating_type,
                    "reason": reason,
                "confidence": round(confidence, 1),
            }
        )

    return results

