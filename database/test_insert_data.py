# import sys, os
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# from database.db_utils import get_session
# from database.db_setup import Student, Classroom, Seat, VerificationLog

# def insert_sample_data():

#     session = get_session()

#     try:
#         # 1) Create classroom
#         classroom = Classroom(
#             name="Classroom A",
#             camera_uri="/dev/video0"
#         )
#         session.add(classroom)
#         session.commit()
#         print("✔ Classroom added")

#         # 2) Add students
#         students = [
#             Student(roll_no=101, name="Arjun", photo_path="assets/student_photos/101.jpg"),
#             Student(roll_no=102, name="Meera", photo_path="assets/student_photos/102.jpg"),
#             Student(roll_no=103, name="Rahul", photo_path="assets/student_photos/103.jpg")
#         ]
#         session.add_all(students)
#         session.commit()
#         print("✔ Students added")

#         # 3) Add seat mapping
#         seats = [
#             Seat(classroom_id=classroom.id, seat_label="Seat-1", bbox="[]", row=1, col=1),
#             Seat(classroom_id=classroom.id, seat_label="Seat-2", bbox="[]", row=1, col=2),
#             Seat(classroom_id=classroom.id, seat_label="Seat-3", bbox="[]", row=1, col=3),
#             Seat(classroom_id=classroom.id, seat_label="Seat-4", bbox="[]", row=2, col=1),
#             Seat(classroom_id=classroom.id, seat_label="Seat-5", bbox="[]", row=2, col=2),
#             Seat(classroom_id=classroom.id, seat_label="Seat-6", bbox="[]", row=2, col=3),
#         ]
#         session.add_all(seats)
#         session.commit()
#         print("✔ Seats added")

#         # 4) Example verification log
#         log = VerificationLog(
#             classroom_id=classroom.id,
#             seat_id=seats[0].id,
#             assigned_roll_no=101,
#             detected_roll_no=101,
#             status="VERIFIED",
#             reason="Face matched",
#             confidence=0.98
#         )
#         session.add(log)
#         session.commit()
#         print("✔ Verification log added")

#     except Exception as e:
#         session.rollback()
#         print("❌ Error:", e)
#     finally:
#         session.close()


# if __name__ == "__main__":
#     insert_sample_data()


# database/insert_seat_mysql.py
import mysql.connector
import json

# --- CONFIG: set your MySQL credentials here ---
MYSQL_CONFIG = {
    "host": "localhost",
    "user": "your_mysql_user",
    "password": "your_mysql_password",
    "database": "exam_monitor",
    "port": 3306
}
# ----------------------------------------------

def insert_seat(classroom_id:int, seat_label:str, bbox:list, row:int, col:int, assigned_roll:int):
    conn = mysql.connector.connect(**MYSQL_CONFIG)
    cur = conn.cursor()
    # If your MySQL supports JSON, use JSON_ARRAY or pass Python JSON and cast
    bbox_json = json.dumps(bbox)
    try:
        # Use parameterized query. If 'bbox' column type is JSON, MySQL will accept the JSON string.
        sql = """INSERT INTO seats (classroom_id, seat_label, bbox, `row`, `col`, assigned_roll_no, status)
                 VALUES (%s, %s, %s, %s, %s, %s, %s)"""
        cur.execute(sql, (classroom_id, seat_label, bbox_json, row, col, assigned_roll, "pending"))
        conn.commit()
        print("Inserted seat id:", cur.lastrowid)
    except Exception as e:
        print("ERROR:", e)
    finally:
        cur.close()
        conn.close()

if __name__ == "__main__":
    # Example: add Seat-1 at bbox [10,20,180,200] assigned to roll 101 in classroom 1
    insert_seat(1, "Seat-1", [10,20,180,200], 1, 1, 101)
