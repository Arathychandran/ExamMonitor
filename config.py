import os

DB_USER = os.getenv("DB_USER", "exam_user")
DB_PASS = os.getenv("DB_PASS", "ExamPass123")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "3306")
DB_NAME = os.getenv("DB_NAME", "EXAM_MONITOR")

SQLALCHEMY_DATABASE_URI = f"mysql+pymysql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}?charset=utf8mb4"

# connection pool settings
POOL_SIZE = int(os.getenv("POOL_SIZE", 10))
MAX_OVERFLOW = int(os.getenv("MAX_OVERFLOW", 5))
POOL_RECYCLE = int(os.getenv("POOL_RECYCLE", 280))