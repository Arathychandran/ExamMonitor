# database/db_setup.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sqlalchemy import (
    create_engine, Column, Integer, String, Text, DateTime, Float, ForeignKey
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker
from sqlalchemy.sql import func
import config

Base = declarative_base()

class Classroom(Base):
    __tablename__ = "classrooms"
    id = Column(Integer, primary_key=True)
    name = Column(String(255), nullable=False, unique=True)
    camera_uri = Column(Text)   # RTSP or device id
    created_at = Column(DateTime, server_default=func.now())
    seats = relationship("Seat", back_populates="classroom", cascade="all, delete-orphan")

class Seat(Base):
    __tablename__ = "seats"
    id = Column(Integer, primary_key=True)
    classroom_id = Column(Integer, ForeignKey("classrooms.id", ondelete="CASCADE"))
    seat_label = Column(String(64))   # e.g., Seat-1 or C1R1
    bbox = Column(Text)               # JSON string: [x,y,w,h]
    row = Column(Integer)
    col = Column(Integer)
    assigned_roll_no = Column(Integer, nullable=True)   # which roll is assigned to this seat
    status = Column(String(32), default="pending")      # pending/verified/mismatch/absent/...
    created_at = Column(DateTime, server_default=func.now())
    classroom = relationship("Classroom", back_populates="seats")

class Student(Base):
    __tablename__ = "students"
    roll_no = Column(Integer, primary_key=True)
    name = Column(String(255))
    photo_path = Column(Text)
    uploaded_by = Column(String(255))
    created_at = Column(DateTime, server_default=func.now())

class VerificationLog(Base):
    __tablename__ = "verification_log"
    id = Column(Integer, primary_key=True)
    classroom_id = Column(Integer, ForeignKey("classrooms.id"))
    seat_id = Column(Integer, ForeignKey("seats.id"))
    assigned_roll_no = Column(Integer)
    detected_roll_no = Column(Integer, nullable=True)
    status = Column(String(32))    # VERIFIED / MISSING / MISMATCH / ERROR
    reason = Column(Text, nullable=True)
    confidence = Column(Float, nullable=True)
    captured_image_path = Column(Text, nullable=True)   # path to evidence image
    timestamp = Column(DateTime, server_default=func.now())

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    username = Column(String(255), unique=True)
    role = Column(String(64))  # admin / teacher
    password_hash = Column(String(255))

def get_engine():
    """
    Create engine from config.SQLALCHEMY_DATABASE_URI.
    config should contain SQLALCHEMY_DATABASE_URI and optionally pool settings.
    """
    engine = create_engine(
        config.SQLALCHEMY_DATABASE_URI,
        pool_size=getattr(config, "POOL_SIZE", 5),
        max_overflow=getattr(config, "MAX_OVERFLOW", 10),
        pool_recycle=getattr(config, "POOL_RECYCLE", 3600),
        echo=False,
        future=True,
    )
    return engine

# session factory helper
_engine = None
_SessionLocal = None

def get_session():
    global _engine, _SessionLocal
    if _engine is None:
        _engine = get_engine()
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(bind=_engine, autoflush=False, autocommit=False, future=True)
    return _SessionLocal()

def create_tables():
    engine = get_engine()
    Base.metadata.create_all(engine)
    print("All tables created.")

if __name__ == "__main__":
    create_tables()
