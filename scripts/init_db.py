"""Create the Postgres schema. Idempotent — safe to re-run.
Usage: python -m scripts.init_db
"""
from app.db import init_schema

if __name__ == "__main__":
    init_schema()
    print("Schema ready (table: chunks).")
