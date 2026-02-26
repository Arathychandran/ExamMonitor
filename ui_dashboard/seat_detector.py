import cv2
import json
import os

class GridSeatDetector:
    def __init__(self, rows=5, cols=6, margin=10):
        """
        rows   : number of seat rows
        cols   : number of seat columns
        margin : padding inside each grid cell
        """
        self.rows = rows
        self.cols = cols
        self.margin = margin

    def detect(self, image_path):
        """
        Detect seat bounding boxes using grid-based logic
        Returns a list of seat dictionaries
        """

        if not os.path.exists(image_path):
            raise FileNotFoundError("Classroom image not found")

        img = cv2.imread(image_path)
        h, w, _ = img.shape

        cell_w = w // self.cols
        cell_h = h // self.rows

        seats = []
        seat_id = 1

        for r in range(self.rows):
            for c in range(self.cols):

                # Grid cell coordinates
                x1 = c * cell_w + self.margin
                y1 = r * cell_h + self.margin
                x2 = (c + 1) * cell_w - self.margin
                y2 = (r + 1) * cell_h - self.margin

                # Safety clamp
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(w, x2)
                y2 = min(h, y2)

                seat = {
                    "seat_label": f"Seat-{seat_id}",
                    "bbox": [x1, y1, x2, y2],
                    "row": r + 1,
                    "col": c + 1
                }

                seats.append(seat)
                seat_id += 1

        return seats

    def save_json(self, classroom_id, camera_uri, seats, output_dir="data"):
        """
        Save detected seats to JSON file
        """

        os.makedirs(output_dir, exist_ok=True)

        seat_map = {
            "classroom_id": classroom_id,
            "camera_uri": camera_uri,
            "seats": seats
        }

        json_path = os.path.join(output_dir, f"seats_{classroom_id}.json")

        with open(json_path, "w") as f:
            json.dump(seat_map, f, indent=2)

        return json_path
