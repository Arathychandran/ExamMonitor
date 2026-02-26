# database/upload_student.py
import os, shutil
from database.db_utils import insert_student
BASE = os.path.join(os.path.dirname(__file__), "..")
PHOTO_DIR = os.path.join(BASE, "assets", "student_photos")
os.makedirs(PHOTO_DIR, exist_ok=True)

def save_student_photo(roll_no:int, name:str, src_path:str)->str:
    ext = os.path.splitext(src_path)[1].lower() or ".jpg"
    dest = os.path.join(PHOTO_DIR, f"roll_{roll_no}{ext}")
    shutil.copy(src_path, dest)
    insert_student(roll_no, name, dest)
    print("Saved", dest)
    return dest

if __name__ == "__main__":
    # Example usage:
    # python database/upload_student.py 101 "Alice" /home/arathy/Pictures/alice.jpg
    import sys
    r = int(sys.argv[1]); n = sys.argv[2]; p = sys.argv[3]
    save_student_photo(r,n,p)
