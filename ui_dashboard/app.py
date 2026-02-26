# from flask import Flask, render_template

# app = Flask(__name__)

# @app.route("/")
# def dashboard():
#     return render_template("dashboard.html")

# if __name__ == "__main__":
#     app.run(debug=True)




# ui_dashboard/app.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import threading
import io
import tempfile
from flask import Flask, render_template, request, redirect, url_for, jsonify, send_from_directory, send_file, Response
from werkzeug.utils import secure_filename
from database.db_setup import create_tables
from database import db_utils
import pandas as pd
import cv2
import numpy as np
import json
from seat_detector import GridSeatDetector
from cheating_detection.cheating_detector import analyze_classroom

# camera coordination: when True for a classroom, MJPEG generator will release the camera and yield blank frames
from collections import defaultdict
camera_pause = defaultdict(lambda: False)
camera_pause_lock = threading.Lock()

# Try import of verifier
from run_verification import run_once

BASE = os.path.join(os.path.dirname(__file__), "..")
PHOTO_DIR = os.path.join(BASE, "assets", "student_photos")
CAP_DIR = os.path.join(BASE, "assets", "captured_faces")
os.makedirs(PHOTO_DIR, exist_ok=True)
os.makedirs(CAP_DIR, exist_ok=True)

ALLOWED = {"png", "jpg", "jpeg"}
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.',1)[1].lower() in ALLOWED

# Serve static files from the project-level "static" folder so uploaded
# classroom images in "static/classrooms" are visible.
app = Flask(__name__, template_folder="templates", static_folder="../static")
create_tables()

# ---------- HOME & CLASSROOMS ----------
@app.route("/")
@app.route("/home")
def home():
    """Home dashboard with KPIs and overview"""
    classrooms = db_utils.get_classrooms()
    classrooms_count = len(classrooms)
    
    # Count students (approximate from student photos directory)
    try:
        import os
        student_photos_dir = os.path.join(BASE, "assets", "student_photos")
        if os.path.exists(student_photos_dir):
            students_count = len([f for f in os.listdir(student_photos_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        else:
            students_count = 0
    except:
        students_count = 0
    
    # Count violations (mismatch + absent from recent logs)
    try:
        violations_count = 0
        suspicious_count = 0
        for c in classrooms:
            logs = db_utils.get_verification_logs_for_classroom(c['id'], limit=100)
            for log in logs:
                if log.get('status') in ('mismatch', 'absent'):
                    violations_count += 1
    except:
        violations_count = 0
        suspicious_count = 0
    
    return render_template(
        "home.html",
        classrooms=classrooms,
        classrooms_count=classrooms_count,
        students_count=students_count,
        violations_count=violations_count,
        suspicious_count=suspicious_count
    )

@app.route("/classrooms")
def index():
    classrooms = db_utils.get_classrooms()
    return render_template("index.html", classrooms=classrooms)

@app.route("/classroom/add", methods=["GET", "POST"])
def add_classroom():
    if request.method == "POST":
        name = request.form.get("name")
        camera_uri = request.form.get("camera_uri")
        num_seats = int(request.form.get("num_seats"))
        image = request.files["classroom_image"]

        classroom_id = db_utils.insert_classroom(name, camera_uri)

        image_path = f"static/classrooms/{classroom_id}.jpg"
        image.save(image_path)

        return redirect(
            url_for(
                "seat_mapping",
                classroom_id=classroom_id,
                num_seats=num_seats
            )
        )

    return render_template("add_classroom.html")


@app.route("/classroom/<int:classroom_id>/seat-mapping")
def seat_mapping(classroom_id):
    num_seats = int(request.args.get("num_seats"))
    image_path = f"classrooms/{classroom_id}.jpg"

    return render_template(
        "seat_mapping.html",
        image=image_path,
        num_seats=num_seats,
        classroom_id=classroom_id
    )


@app.route("/classroom/<int:classroom_id>/save-seats", methods=["POST"])
def save_seats(classroom_id):
    data = request.json
    camera_uri = db_utils.get_camera_uri(classroom_id)

    # 1) Save raw seats (using [x1,y1,x2,y2]) into data/seats_<id>.json for the app.
    seat_json = {
        "classroom_id": classroom_id,
        "camera_uri": camera_uri,
        "seats": data["seats"],
    }
    with open(f"data/seats_{classroom_id}.json", "w") as f:
        json.dump(seat_json, f, indent=2)

    # 2) Also update seat_detection/seat_config_classroom_<id>.json
    #    converting bbox to [x, y, w, h] as used there.
    try:
        seat_cfg_path = os.path.join(BASE, "seat_detection", f"seat_config_classroom_{classroom_id}.json")
        existing = {}
        cfg = {}
        if os.path.exists(seat_cfg_path):
            with open(seat_cfg_path, "r") as f:
                cfg = json.load(f) or {}
            for s in cfg.get("seats", []):
                label = s.get("seat_label")
                if label:
                    existing[label] = s
        else:
            cfg = {"classroom_id": classroom_id, "camera_uri": camera_uri, "seats": []}

        incoming = {s["seat_label"]: s for s in data["seats"]}
        new_seats = []

        # Update existing seats by label
        for label, s in existing.items():
            if label in incoming:
                b = incoming[label]["bbox"]
                x1, y1, x2, y2 = b[0], b[1], b[2], b[3]
                s["bbox"] = [x1, y1, x2 - x1, y2 - y1]
            new_seats.append(s)

        # Add any new labels that weren't present before
        for label, seat in incoming.items():
            if label in existing:
                continue
            b = seat["bbox"]
            x1, y1, x2, y2 = b[0], b[1], b[2], b[3]
            new_seats.append(
                {
                    "seat_label": label,
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "row": None,
                    "col": None,
                    "assigned_roll_no": None,
                }
            )

        cfg["classroom_id"] = classroom_id
        cfg["camera_uri"] = camera_uri
        cfg["seats"] = new_seats

        with open(seat_cfg_path, "w") as f:
            json.dump(cfg, f, indent=2)
    except Exception as e:
        print("[save_seats] warning: could not update seat_config json:", e)

    # 3) Also export an Excel sheet summarizing the seats for this classroom.
    #    This is mainly for your reference; the app itself still reads JSON.
    try:
        import pandas as pd
        rows = []
        for seat in data["seats"]:
            b = seat["bbox"]
            x1, y1, x2, y2 = b[0], b[1], b[2], b[3]
            rows.append(
                {
                    "Seat_Label": seat.get("seat_label"),
                    "X1": x1,
                    "Y1": y1,
                    "X2": x2,
                    "Y2": y2,
                    "Assigned_Roll_No": seat.get("assigned_roll_no"),
                }
            )
        df = pd.DataFrame(rows)
        os.makedirs(os.path.join(BASE, "data"), exist_ok=True)
        xls_path = os.path.join(BASE, "data", f"seats_{classroom_id}.xlsx")
        df.to_excel(xls_path, index=False)
        print(f"[save_seats] Excel written to {xls_path}")
    except Exception as e:
        print("[save_seats] warning: could not write Excel:", e)

    return {"status": "success"}


# ---------- EDIT SEATING (XLS-style UI) ----------
@app.route("/classroom/<int:classroom_id>/seating", methods=["GET", "POST"])
def edit_seating(classroom_id):
    """
    Display the seats_<id>.xlsx sheet as an HTML table so teachers can
    assign roll numbers per seat, then push those assignments back to
    the JSON config and database.
    """
    import pandas as pd

    classroom = db_utils.get_classroom(classroom_id)
    if not classroom:
        return "Classroom not found", 404

    xls_path = os.path.join(BASE, "data", f"seats_{classroom_id}.xlsx")
    json_path = os.path.join(BASE, "data", f"seats_{classroom_id}.json")

    if request.method == "POST":
        labels = request.form.getlist("seat_label")
        rolls = request.form.getlist("assigned_roll_no")
        mapping = {}
        for label, roll in zip(labels, rolls):
            roll = roll.strip()
            if roll == "":
                mapping[label] = None
            else:
                try:
                    mapping[label] = int(roll)
                except ValueError:
                    mapping[label] = roll  # leave as string if not numeric

        # Update Excel
        try:
            if os.path.exists(xls_path):
                df = pd.read_excel(xls_path)
            elif os.path.exists(json_path):
                with open(json_path, "r") as f:
                    data = json.load(f)
                rows = []
                for s in data.get("seats", []):
                    b = s.get("bbox", [0, 0, 0, 0])
                    rows.append(
                        {
                            "Seat_Label": s.get("seat_label"),
                            "X1": b[0],
                            "Y1": b[1],
                            "X2": b[2],
                            "Y2": b[3],
                            "Assigned_Roll_No": s.get("assigned_roll_no"),
                        }
                    )
                df = pd.DataFrame(rows)
            else:
                df = pd.DataFrame(columns=["Seat_Label", "X1", "Y1", "X2", "Y2", "Assigned_Roll_No"])

            if not df.empty and "Seat_Label" in df.columns:
                df["Assigned_Roll_No"] = df["Seat_Label"].map(mapping)
            df.to_excel(xls_path, index=False)
        except Exception as e:
            print("[edit_seating] warning updating Excel:", e)

        # Update seat_config JSON and DB seats
        try:
            cfg_path = os.path.join(BASE, "seat_detection", f"seat_config_classroom_{classroom_id}.json")
            if os.path.exists(cfg_path):
                with open(cfg_path, "r") as f:
                    cfg = json.load(f) or {}
            else:
                cfg = {"classroom_id": classroom_id, "camera_uri": classroom.get("camera_uri"), "seats": []}

            for s in cfg.get("seats", []):
                label = s.get("seat_label")
                if label in mapping:
                    s["assigned_roll_no"] = mapping[label]

            with open(cfg_path, "w") as f:
                json.dump(cfg, f, indent=2)

            # Push to DB using existing import helper
            try:
                from excel_manager.import_seating import import_from_seat_config
                import_from_seat_config(cfg_path)
            except Exception as e:
                print("[edit_seating] warning importing into DB:", e)
        except Exception as e:
            print("[edit_seating] warning updating seat_config:", e)

        # Ensure Student records exist for each assigned roll number using photos
        # from assets/student_photos. We support both "roll_<roll>.ext" and
        # "<roll>.ext" naming.
        try:
            for label, roll in mapping.items():
                if roll is None:
                    continue
                # Only handle simple integer roll numbers here
                try:
                    roll_int = int(roll)
                except Exception:
                    continue

                photo_path = None
                for ext in ("jpg", "jpeg", "png"):
                    # look for roll_<roll>.ext
                    cand1 = os.path.join(PHOTO_DIR, f"roll_{roll_int}.{ext}")
                    # or plain <roll>.ext (as in "103.jpeg")
                    cand2 = os.path.join(PHOTO_DIR, f"{roll_int}.{ext}")
                    if os.path.exists(cand1):
                        photo_path = cand1
                        break
                    if os.path.exists(cand2):
                        photo_path = cand2
                        break

                if photo_path:
                    # Name is optional here; can be filled later.
                    db_utils.insert_student(roll_int, name=f"Student {roll_int}", photo_path=photo_path)
                else:
                    print(f"[edit_seating] no photo found for roll {roll_int} in {PHOTO_DIR}")
        except Exception as e:
            print("[edit_seating] warning ensuring student photos:", e)

        return redirect(url_for("classroom_detail", classroom_id=classroom_id))

    # GET: load rows to display
    rows = []
    try:
        if os.path.exists(xls_path):
            df = pd.read_excel(xls_path)
        elif os.path.exists(json_path):
            with open(json_path, "r") as f:
                data = json.load(f)
            tmp = []
            for s in data.get("seats", []):
                b = s.get("bbox", [0, 0, 0, 0])
                tmp.append(
                    {
                        "Seat_Label": s.get("seat_label"),
                        "X1": b[0],
                        "Y1": b[1],
                        "X2": b[2],
                        "Y2": b[3],
                        "Assigned_Roll_No": s.get("assigned_roll_no"),
                    }
                )
            df = pd.DataFrame(tmp)
        else:
            df = pd.DataFrame(columns=["Seat_Label", "X1", "Y1", "X2", "Y2", "Assigned_Roll_No"])

        if not df.empty:
            rows = df.to_dict(orient="records")
    except Exception as e:
        print("[edit_seating] warning reading seats:", e)

    return render_template("edit_seating.html", classroom=classroom, rows=rows)


# ---------- CHEATING SUMMARY API ----------
@app.route("/api/cheating_summary/<int:classroom_id>", methods=["POST"])
def api_cheating_summary(classroom_id):
    """
    Run an offline cheating-analysis pass for this classroom and return
    a per-seat summary:
      { seat_label, assigned_roll_no, status, cheating_type, confidence }
    """
    classroom = db_utils.get_classroom(classroom_id)
    if not classroom:
        return jsonify({"ok": False, "error": "classroom not found"}), 404

    camera_uri = db_utils.get_classroom_camera(classroom_id)
    import time

    try:
        # Pause MJPEG streaming so cheating analysis can open the camera,
        # just like we do for verification.
        with camera_pause_lock:
            camera_pause[classroom_id] = True
        time.sleep(0.8)

        summary = analyze_classroom(
            classroom_id=classroom_id,
            camera_source=camera_uri,
            duration_seconds=10,
            target_fps=5,
        )

        if not summary:
            # If we couldn't get any data (e.g., camera busy/unavailable),
            # signal an error to the frontend instead of silently succeeding.
            return jsonify({"ok": False, "error": "Cheating analysis had no data; camera may be unavailable."}), 200

        return jsonify({"ok": True, "results": summary})
    except Exception as e:
        print("[api_cheating_summary] error:", e)
        return jsonify({"ok": False, "error": str(e)}), 500
    finally:
        # Always resume MJPEG streaming
        with camera_pause_lock:
            camera_pause[classroom_id] = False


# ---------- CLASSROOM DETAIL ----------
@app.route("/classroom/<int:classroom_id>")
def classroom_detail(classroom_id):
    classroom = db_utils.get_classroom(classroom_id)
    seats = db_utils.get_seats_by_classroom(classroom_id)
    return render_template("classroom.html", classroom=classroom, seats=seats)

# ---------- UPLOAD STUDENT ----------
@app.route("/upload_student", methods=["GET","POST"])
def upload_student():
    if request.method == "POST":
        roll_no = int(request.form.get("roll_no"))
        name = request.form.get("name", "")
        file = request.files.get("file")
        if not file or file.filename == "":
            return "No file", 400
        if not allowed_file(file.filename):
            return "Invalid file type", 400
        ext = secure_filename(file.filename).rsplit('.',1)[1]
        dest_name = f"roll_{roll_no}.{ext}"
        dest_path = os.path.join(PHOTO_DIR, dest_name)
        file.save(dest_path)
        db_utils.insert_student(roll_no, name, dest_path)
        return redirect(url_for("index"))
    return render_template("upload_student.html")

# ---------- IMPORT SEATING ----------
@app.route("/import_seating", methods=["GET","POST"])
def import_seating():
    if request.method == "POST":
        file = request.files.get("file")
        if not file:
            return "No file", 400
        fname = secure_filename(file.filename)
        tmp = os.path.join(tempfile.gettempdir(), fname)
        file.save(tmp)
        from excel_manager.import_seating import import_from_seat_config
        import_from_seat_config(tmp)
        return redirect(url_for("index"))
    return render_template("import_seating.html")

# ---------- START VERIFICATION (background) ----------
verification_threads = {}
# allow both GET and POST so a direct browser navigation works during debugging
@app.route("/start_verification/<int:classroom_id>", methods=["GET", "POST"])
def start_verification(classroom_id):
    print(f"[start_verification] HTTP request received for classroom {classroom_id} (method={request.method})")
    classroom = db_utils.get_classroom(classroom_id)
    if not classroom:
        return jsonify({"error":"classroom not found"}), 404
    camera_uri = db_utils.get_classroom_camera(classroom_id)

    def worker():
        try:
            # request pause on MJPEG stream for this classroom
            with camera_pause_lock:
                camera_pause[classroom_id] = True
            print(f"[start_verification] Camera pause requested for classroom {classroom_id}")
            # give the streaming thread a moment to release camera
            import time
            time.sleep(0.8)

            print(f"[start_verification] Worker starting for classroom {classroom_id} using camera {camera_uri}")
            run_once(classroom_id, camera_uri)
            print(f"[start_verification] Worker finished for classroom {classroom_id}")
        except Exception as e:
            print("[start_verification] error:", e)
        finally:
            # remove pause to resume MJPEG streaming
            with camera_pause_lock:
                camera_pause[classroom_id] = False
            print(f"[start_verification] Camera pause cleared for classroom {classroom_id}")

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    verification_threads[classroom_id] = t

    if request.method == "POST":
        return jsonify({"ok": True, "message": "verification started"})
    else:
        return redirect(url_for("classroom_detail", classroom_id=classroom_id))

# DEBUG helper: runs verification in foreground and returns a quick summary (for debugging only)
@app.route("/debug/run_once/<int:classroom_id>")
def debug_run_once(classroom_id):
    classroom = db_utils.get_classroom(classroom_id)
    if not classroom:
        return "classroom not found", 404
    camera_uri = db_utils.get_classroom_camera(classroom_id)
    try:
        print(f"[debug_run_once] Running synchronous run_once for classroom {classroom_id} camera {camera_uri}")
        run_once(classroom_id, camera_uri)
        # after run, return number of logs inserted for that classroom (quick check)
        logs = db_utils.get_verification_logs_for_classroom(classroom_id, limit=10000)
        return jsonify({"ok": True, "logs_returned": len(logs)})
    except Exception as e:
        print("[debug_run_once] error:", e)
        return jsonify({"ok": False, "error": str(e)}), 500


# ---------- MJPEG video feed for classroom ----------
def gen_mjpeg(camera_source, classroom_id=None):
    """Yield MJPEG frames from camera_source (int or rtsp).
    If camera_pause[classroom_id] is True, release the camera and yield blank frames
    until pause is cleared. This allows verification to open the camera exclusively.
    """
    import cv2, time, numpy as np

    def open_cap(source):
        try:
            if isinstance(source, str) and source.isdigit():
                return cv2.VideoCapture(int(source))
            return cv2.VideoCapture(source)
        except Exception:
            return cv2.VideoCapture(source)

    # Try to open
    cap = open_cap(camera_source)
    if not cap or not cap.isOpened():
        # If pause requested immediately, just yield blanks and exit
        if classroom_id is not None and camera_pause.get(classroom_id, False):
            blank = 255 * np.ones((480, 640, 3), dtype='uint8')
            ret, buf = cv2.imencode('.jpg', blank)
            frame_bytes = buf.tobytes()
            while camera_pause.get(classroom_id, False):
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            # attempt reopen after pause ends
            cap = open_cap(camera_source)

    # Stream loop with pause check
    while True:
        # If pause requested, release current capture and yield blanks until resume
        if classroom_id is not None and camera_pause.get(classroom_id, False):
            try:
                if cap and cap.isOpened():
                    cap.release()
            except Exception:
                pass
            blank = 255 * np.ones((480, 640, 3), dtype='uint8')
            ret, buf = cv2.imencode('.jpg', blank)
            frame_bytes = buf.tobytes()
            # yield blanks until pause lifted
            while camera_pause.get(classroom_id, False):
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            # after pause ends, try to re-open
            cap = open_cap(camera_source)
            if not cap or not cap.isOpened():
                # if still can't open, continue yielding blanks
                while not (cap and cap.isOpened()):
                    yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                    time.sleep(0.5)

        # Normal frame read
        if not cap:
            cap = open_cap(camera_source)
        if not cap or not cap.isOpened():
            # yield blank frame if camera unavailable
            blank = 255 * np.ones((480, 640, 3), dtype='uint8')
            ret, buf = cv2.imencode('.jpg', blank)
            if not ret:
                time.sleep(0.1)
                continue
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')
            time.sleep(0.2)
            continue

        ret, frame = cap.read()
        if not ret or frame is None:
            # short wait and retry
            time.sleep(0.1)
            continue

        # optional resize
        h, w = frame.shape[:2]
        if w > 1000:
            frame = cv2.resize(frame, (1000, int(1000 * h / w)))
        ret2, jpeg = cv2.imencode('.jpg', frame)
        if not ret2:
            continue
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')

    # never reached - but safe release
    try:
        if cap and cap.isOpened():
            cap.release()
    except Exception:
        pass

@app.route('/video_feed/<int:classroom_id>')
def video_feed(classroom_id):
    camera_uri = db_utils.get_classroom_camera(classroom_id)
    return Response(gen_mjpeg(camera_uri, classroom_id=classroom_id),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# ---------- API: seats + logs ----------
@app.route("/api/seats/<int:classroom_id>")
def api_seats(classroom_id):
    seats = db_utils.get_seats_by_classroom(classroom_id)
    return jsonify([s for s in seats])

@app.route("/api/logs/seat/<int:seat_id>")
def api_logs(seat_id):
    logs = db_utils.get_logs_for_seat(seat_id)
    return jsonify(logs)

# ---------- API: manual override ----------
@app.route("/api/seat/override", methods=["POST"])
def api_seat_override():
    data = request.json or request.form
    seat_id = int(data.get("seat_id"))
    new_status = data.get("status")
    reason = data.get("reason", "manual_override")
    classroom_id = int(data.get("classroom_id"))
    assigned_roll = data.get("assigned_roll_no", None)
    detected_roll = data.get("detected_roll_no", None)
    # update seat and insert log
    db_utils.update_seat_status(seat_id, new_status)
    db_utils.insert_verification_log(classroom_id, seat_id, f"seat_{seat_id}", assigned_roll, detected_roll, new_status, reason, None, None)
    return jsonify({"ok":True})

# ---------- REPORT DOWNLOAD (Excel) ----------
@app.route("/reports/download/<int:classroom_id>")
def download_report(classroom_id):
    """
    Build an Excel report from verification_log rows using SQLAlchemy (db_utils).
    """
    import pandas as pd, tempfile, os
    # Use the helper that returns verification logs for the classroom
    try:
        rows = db_utils.get_verification_logs_for_classroom(classroom_id, limit=10000)
    except Exception as e:
        print("[download_report] error fetching logs:", e)
        rows = []

    # Build a compact Excel report with the exact columns requested:
    # Seat Number, Roll Number, Reference Photo, Captured Photo, Output (match/mismatch/absent)
    import os

    if rows and isinstance(rows, list):
        df_raw = pd.DataFrame(rows)
    else:
        df_raw = pd.DataFrame(columns=[
            "seat_label", "assigned_roll_no", "status", "reason", "captured_image_path"
        ])

    # Ensure required columns exist
    for c in ["seat_label", "assigned_roll_no", "status", "reason", "captured_image_path"]:
        if c not in df_raw.columns:
            df_raw[c] = None

    # Resolve reference photo path for each roll number
    from database import db_utils as _dbu
    ref_photos = {}
    for roll in df_raw["assigned_roll_no"].dropna().unique():
        try:
            roll_int = int(roll)
        except Exception:
            continue
        student = _dbu.get_student(roll_int)
        path = None
        if student and student.get("photo_path"):
            path = student["photo_path"]
        else:
            # best-effort guess based on naming convention
            base = os.path.join(BASE, "assets", "student_photos")
            for ext in ("jpg", "jpeg", "png"):
                cand1 = os.path.join(base, f"roll_{roll_int}.{ext}")
                cand2 = os.path.join(base, f"{roll_int}.{ext}")
                if os.path.exists(cand1):
                    path = cand1
                    break
                if os.path.exists(cand2):
                    path = cand2
                    break
        ref_photos[roll_int] = path

    def map_output(row):
        st = (row.get("status") or "").lower()
        if st == "verified":
            return "match"
        if st == "mismatch":
            return "mismatch"
        if st == "absent":
            return "absent"
        if "missing_photo" in st:
            return "absent"
        rs = (row.get("reason") or "").lower()
        if "no_face" in rs:
            return "absent"
        return st or "unknown"

    report_df = pd.DataFrame({
        "Seat_Number": df_raw["seat_label"],
        "Roll_Number": df_raw["assigned_roll_no"],
        "Reference_Photo": df_raw["assigned_roll_no"].apply(lambda r: ref_photos.get(int(r)) if pd.notna(r) and str(r).isdigit() else None),
        "Captured_Photo": df_raw["captured_image_path"],
        "Output": df_raw.apply(map_output, axis=1),
    })

    out_path = os.path.join(tempfile.gettempdir(), f"report_classroom_{classroom_id}.xlsx")
    report_df.to_excel(out_path, index=False)
    return send_file(out_path, as_attachment=True, download_name=f"report_classroom_{classroom_id}.xlsx")

# ---------- Serve images ----------
@app.route("/images/student/<path:filename>")
def student_image(filename):
    return send_from_directory(PHOTO_DIR, filename)

@app.route("/images/captured/<path:filename>")
def captured_image(filename):
    return send_from_directory(CAP_DIR, filename)

# ---- DRAW UI page ----
@app.route("/classroom/<int:classroom_id>/draw")
def classroom_draw(classroom_id):
    classroom = db_utils.get_classroom(classroom_id)
    if not classroom:
        return "Classroom not found", 404
    # get existing seats to draw them on canvas
    seats = db_utils.get_seats_by_classroom(classroom_id)
    return render_template("draw_bboxes.html", classroom=classroom, seats=seats)

# ---- API: save seat bbox ----
@app.route("/api/save_seat", methods=["POST"])
def api_save_seat():
    data = request.get_json() or {}
    try:
        classroom_id = int(data.get("classroom_id"))
        seat_label = data.get("seat_label") or "seat"
        bbox = data.get("bbox")  # expected [x,y,w,h] absolute pixel values for the camera frame
        row = data.get("row", None)
        col = data.get("col", None)
        assigned_roll_no = data.get("assigned_roll_no", None)
        # Normalize bbox to JSON string for DB insert_or_update_seat
        import json
        bbox_json = json.dumps(bbox)
        sid = db_utils.insert_or_update_seat(classroom_id, seat_label, bbox_json, row, col, assigned_roll_no)
        return jsonify({"ok": True, "seat_id": sid})
    except Exception as e:
        print("[api_save_seat] error:", e)
        return jsonify({"ok": False, "error": str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
