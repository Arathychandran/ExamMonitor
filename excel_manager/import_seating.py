# excel_manager/import_seating.py
import json
import os
from typing import Any, Dict, List
from database import db_utils

def import_from_seat_config(path: str) -> Dict[str, Any]:
    """
    Import seats from a JSON seat config file.

    Expected JSON format:
    {
      "classroom_id": 2,
      "camera_uri": "0",
      "seats": [
         {"seat_label":"Seat-1","bbox":[50,50,200,200],"row":1,"col":1,"assigned_roll_no":101},
         ...
      ]
    }

    The function upserts seats into the seats table using db_utils.insert_or_update_seat.
    Returns a summary dict.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Seat config not found: {path}")

    with open(path, "r") as f:
        data = json.load(f)

    classroom_id = data.get("classroom_id")
    if classroom_id is None:
        raise ValueError("classroom_id is required in seat config JSON")

    # optional: update classroom camera URI if present
    camera_uri = data.get("camera_uri", None)
    if camera_uri is not None:
        try:
            # update classroom record if exists
            cls = db_utils.get_classroom(classroom_id)
            if cls:
                # Use SQLAlchemy helper to update - we don't have a dedicated update function,
                # so re-insert via delete/insert could be heavy. We'll use insert_classroom only if not exists.
                pass
        except Exception:
            pass

    seats: List[Dict[str, Any]] = data.get("seats", [])
    inserted = 0
    updated = 0
    for s in seats:
        seat_label = s.get("seat_label")
        bbox = s.get("bbox")  # list or string
        row = s.get("row")
        col = s.get("col")
        assigned = s.get("assigned_roll_no", None)
        if seat_label is None or bbox is None:
            continue
        # store bbox as JSON string (db_utils expects a string)
        bbox_text = json.dumps(bbox) if not isinstance(bbox, str) else bbox
        seat_id = db_utils.insert_or_update_seat(classroom_id, seat_label, bbox_text, row, col, assigned)
        inserted += 1

    return {"ok": True, "classroom_id": classroom_id, "seats_processed": inserted}
