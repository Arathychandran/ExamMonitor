# database/db_utils.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import os, json
from typing import Optional, List, Dict, Any
from sqlalchemy import create_engine, select, update
from sqlalchemy.orm import sessionmaker
from sqlalchemy.orm import Session
from config import SQLALCHEMY_DATABASE_URI
from database.db_setup import Base, Classroom, Seat, Student, VerificationLog, User


engine = create_engine(SQLALCHEMY_DATABASE_URI, pool_pre_ping=True, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)

# --- Helper: ensure tables exist (optional) ---
def ensure_tables():
    Base.metadata.create_all(engine)


# -------------------------
# Helpers
# -------------------------
def _row_to_dict(obj) -> Dict[str, Any]:
    """Convert a SQLAlchemy ORM object to dict (shallow)."""
    if obj is None:
        return None
    d = {}
    for c in obj.__table__.columns:
        d[c.name] = getattr(obj, c.name)
    return d

# -------------------------
# CLASSROOMS
# -------------------------
def get_classrooms() -> List[Dict[str, Any]]:
    """Return list of all classrooms."""
    with SessionLocal() as session:
        stmt = select(Classroom).order_by(Classroom.id.desc())
        rows = session.execute(stmt).scalars().all()
        return [_row_to_dict(r) for r in rows]

def insert_classroom(name: str, camera_uri: str) -> int:
    """Insert a new classroom and return its id."""
    with SessionLocal() as session:
        c = Classroom(name=name, camera_uri=camera_uri)
        session.add(c)
        session.commit()
        session.refresh(c)
        return c.id

def get_classroom(classroom_id: int) -> Optional[Dict[str, Any]]:
    with SessionLocal() as session:
        stmt = select(Classroom).where(Classroom.id == classroom_id)
        r = session.execute(stmt).scalar_one_or_none()
        return _row_to_dict(r)

def get_classroom_camera(classroom_id: int) -> str:
    with SessionLocal() as session:
        stmt = select(Classroom.camera_uri).where(Classroom.id == classroom_id)
        r = session.execute(stmt).scalar_one_or_none()
        return str(r) if r is not None else "0"

def get_camera_uri(classroom_id: int) -> str:
    with SessionLocal() as session:
        stmt = select(Classroom.camera_uri).where(Classroom.id == classroom_id)
        camera_uri = session.execute(stmt).scalar_one_or_none()

        return camera_uri if camera_uri is not None else "0"


# -------------------------
# STUDENTS
# -------------------------
def insert_student(roll_no: int, name: str, photo_path: str):
    """Insert or replace a student record."""
    with SessionLocal() as session:
        # Try to get existing
        stmt = select(Student).where(Student.roll_no == roll_no)
        existing = session.execute(stmt).scalar_one_or_none()
        if existing:
            existing.name = name
            existing.photo_path = photo_path
            session.add(existing)
        else:
            s = Student(roll_no=roll_no, name=name, photo_path=photo_path)
            session.add(s)
        session.commit()

def get_student(roll_no: int) -> Optional[Dict[str, Any]]:
    with SessionLocal() as session:
        stmt = select(Student).where(Student.roll_no == roll_no)
        r = session.execute(stmt).scalar_one_or_none()
        return _row_to_dict(r)


# -------------------------
# SEATS
# -------------------------
# database/db_utils.py
from typing import List, Dict, Any
def get_seats_by_classroom(classroom_id: int) -> List[Dict[str, Any]]:
    """
    SQLAlchemy ORM implementation using the bound SessionLocal factory.
    Returns list of seat dicts ordered by col then row.
    """
    with SessionLocal() as session:
        rows = session.query(Seat).filter(Seat.classroom_id == classroom_id).order_by(Seat.col.asc(), Seat.row.asc()).all()

        result = []
        for obj in rows:
            row = {}
            for c in obj.__table__.columns:
                row[c.name] = getattr(obj, c.name)
            # if bbox stored as TEXT/JSON, try to parse
            if 'bbox' in row and isinstance(row['bbox'], str):
                try:
                    row['bbox'] = json.loads(row['bbox'])
                except Exception:
                    pass
            result.append(row)
        return result

def update_seat_status(seat_id: int, status: str):
    with SessionLocal() as session:
        stmt = select(Seat).where(Seat.id == seat_id)
        s = session.execute(stmt).scalar_one_or_none()
        if s:
            s.status = status
            session.add(s)
            session.commit()

def insert_or_update_seat(classroom_id: int, seat_label: str, bbox: str, row: Optional[int], col: Optional[int], assigned_roll_no: Optional[int]):
    """
    Upsert seat by (classroom_id, seat_label).
    bbox should be a JSON string like "[x,y,w,h]" or plain string.
    """
    with SessionLocal() as session:
        stmt = select(Seat).where(Seat.classroom_id == classroom_id, Seat.seat_label == seat_label)
        existing = session.execute(stmt).scalar_one_or_none()
        if existing:
            existing.bbox = bbox
            existing.row = row
            existing.col = col
            existing.assigned_roll_no = assigned_roll_no
            session.add(existing)
            session.commit()
            return existing.id
        else:
            s = Seat(classroom_id=classroom_id, seat_label=seat_label, bbox=bbox, row=row, col=col, assigned_roll_no=assigned_roll_no, status="pending")
            session.add(s)
            session.commit()
            session.refresh(s)
            return s.id


# -------------------------
# VERIFICATION LOGS
# -------------------------
# Replace existing insert_verification_log in database/db_utils.py with this robust version

def insert_verification_log(classroom_id: int, seat_id: int, seat_label: str,
                            assigned_roll_no: Optional[int], detected_roll_no: Optional[int],
                            status: str, reason: Optional[str], confidence: Optional[float],
                            captured_image_path: Optional[str]):
    """
    Insert a verification log row into the DB.
    This function is defensive: it only passes keyword args that the VerificationLog model actually defines,
    so it won't crash if the model schema differs (for example, if `seat_label` is not a column).
    """
    with SessionLocal() as session:
        # Build candidate kwargs
        candidate = {
            "classroom_id": classroom_id,
            "seat_id": seat_id,
            "seat_label": seat_label,
            "assigned_roll_no": assigned_roll_no,
            "detected_roll_no": detected_roll_no,
            "status": status,
            "reason": reason,
            "confidence": confidence,
            "captured_image_path": captured_image_path
        }

        # Inspect model columns and filter candidate to only allowed keys
        allowed_cols = {c.name for c in VerificationLog.__table__.columns}
        filtered = {k: v for k, v in candidate.items() if k in allowed_cols}

        # Create log row with filtered args
        log = VerificationLog(**filtered)
        session.add(log)
        session.commit()


def get_logs_for_seat(seat_id: int, limit: int = 20) -> List[Dict[str, Any]]:
    with SessionLocal() as session:
        stmt = select(VerificationLog).where(VerificationLog.seat_id == seat_id).order_by(VerificationLog.timestamp.desc()).limit(limit)
        rows = session.execute(stmt).scalars().all()
        return [_row_to_dict(r) for r in rows]


# -------------------------
# Additional convenience helpers (optional)
# -------------------------
def delete_classroom(classroom_id: int):
    with SessionLocal() as session:
        stmt = select(Classroom).where(Classroom.id == classroom_id)
        c = session.execute(stmt).scalar_one_or_none()
        if c:
            # Optionally cascade delete seats/logs if you want
            # session.query(Seat).filter(Seat.classroom_id==classroom_id).delete()
            session.delete(c)
            session.commit()

def get_verification_logs_for_classroom(classroom_id: int, limit: int = 100):
    with SessionLocal() as session:
        stmt = select(VerificationLog).where(VerificationLog.classroom_id == classroom_id).order_by(VerificationLog.timestamp.desc()).limit(limit)
        rows = session.execute(stmt).scalars().all()
        return [_row_to_dict(r) for r in rows]
