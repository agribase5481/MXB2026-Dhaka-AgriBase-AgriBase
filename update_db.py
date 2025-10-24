"""
Create/Populate trial.db from CSVs in extracted_tables_2024.
- Preserves exact CSV rows as JSON (header->value) to keep original cell layout.
- Detects variety presence per table and records tables_meta (filename, page, crop, descriptor, scope, has_variety).
- Does NOT touch agri-base.db (creates/uses trial.db).
Run:
    python3 update_trial_db.py --csvdir extracted_tables_2024 --db trial.db
"""
import argparse
import sqlite3
import json
import re
from pathlib import Path
from datetime import datetime

import pandas as pd

DESCRIPTOR_TOKENS = {
    "area", "production", "prod", "area_prod", "yield", "yield_rate", "price",
    "expenditure", "estimate", "productivity", "cost"
}
SCOPE_TOKENS = {"dist", "div", "upz", "thana", "district", "division", "upazila"}


def parse_filename(fname: str):
    stem = Path(fname).stem
    parts = stem.split('_')
    page = None
    crop = None
    descriptor = None
    scope = None

    # page prefix
    if parts and re.fullmatch(r'\d+', parts[0]):
        page = int(parts[0])
        parts = parts[1:]

    # last tokens may be scope/descriptor
    if parts:
        if parts[-1].lower() in SCOPE_TOKENS:
            scope = parts.pop(-1).lower()
    if parts:
        if parts[-1].lower() in DESCRIPTOR_TOKENS:
            descriptor = parts.pop(-1).lower()
    # handle compound 'area_prod'
    if parts and re.search(r'area[_\-]?prod', parts[-1].lower()):
        descriptor = 'area_prod'
        parts.pop(-1)

    if parts:
        crop = "_".join(parts)
        crop = re.sub(r'[^A-Za-z0-9_]', '', crop).strip('_')
        if crop == '':
            crop = None
    return page, crop, descriptor, scope


def detect_variety(df: pd.DataFrame) -> bool:
    cols = [str(c).lower() for c in df.columns]
    if any('variet' in c or 'variety' in c for c in cols):
        return True
    # sample values
    sample = df.head(80).astype(str).applymap(lambda x: x.lower() if pd.notnull(x) else '')
    if sample.apply(lambda col: col.str.contains(r'variet|variety|hybrid|local', na=False)).any().any():
        return True
    return False


def row_to_dict(header, row):
    out = {}
    # ensure header list is aligned
    for i, val in enumerate(row):
        key = header[i] if i < len(header) else f"col_{i}"
        if pd.isna(key) or str(key).strip() == "":
            key = f"col_{i}"
        out[str(key).strip()] = "" if pd.isna(val) else str(val)
    return out


def ensure_schema(conn: sqlite3.Connection):
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS crops (
        id INTEGER PRIMARY KEY,
        name TEXT UNIQUE
    )""")
    cur.execute("""
    CREATE TABLE IF NOT EXISTS tables_meta (
        id INTEGER PRIMARY KEY,
        filename TEXT UNIQUE,
        page INTEGER,
        crop TEXT,
        descriptor TEXT,
        scope TEXT,
        has_variety INTEGER,
        created_at TEXT
    )""")
    cur.execute("""
    CREATE TABLE IF NOT EXISTS crop_data (
        id INTEGER PRIMARY KEY,
        table_id INTEGER,
        row_index INTEGER,
        row_json TEXT,
        FOREIGN KEY(table_id) REFERENCES tables_meta(id)
    )""")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_tables_crop ON tables_meta(crop)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_data_table ON crop_data(table_id)")
    conn.commit()


def ingest_csv(conn: sqlite3.Connection, csv_path: Path, overwrite=False):
    fname = csv_path.name
    page, crop, descriptor, scope = parse_filename(fname)
    # read CSV robustly
    try:
        df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    except Exception:
        df = pd.read_csv(csv_path, dtype=str, encoding='utf-8', engine='python', keep_default_na=False, on_bad_lines='skip')

    header = list(df.columns)
    has_var = 1 if detect_variety(df) else 0

    cur = conn.cursor()
    if crop:
        cur.execute("INSERT OR IGNORE INTO crops(name) VALUES(?)", (crop,))
    now = datetime.utcnow().isoformat()
    if overwrite:
        cur.execute("SELECT id FROM tables_meta WHERE filename = ?", (fname,))
        r = cur.fetchone()
        if r:
            tid = r[0]
            cur.execute("DELETE FROM crop_data WHERE table_id = ?", (tid,))
            cur.execute("DELETE FROM tables_meta WHERE id = ?", (tid,))
            conn.commit()

    cur.execute("""
        INSERT OR IGNORE INTO tables_meta(filename, page, crop, descriptor, scope, has_variety, created_at)
        VALUES(?,?,?,?,?,?,?)""",
        (fname, page, crop, descriptor, scope, has_var, now))
    conn.commit()
    cur.execute("SELECT id FROM tables_meta WHERE filename = ?", (fname,))
    table_id = cur.fetchone()[0]

    # store rows
    for idx, row in enumerate(df.values.tolist()):
        row_obj = row_to_dict(header, row)
        cur.execute("INSERT INTO crop_data(table_id, row_index, row_json) VALUES(?,?,?)",
                    (table_id, idx, json.dumps(row_obj, ensure_ascii=False)))
    conn.commit()
    return table_id, has_var


def main():
    parser = argparse.ArgumentParser(description="Import CSVs into trial.db (safe, non-destructive)")
    parser.add_argument("--db", default="trial.db", help="SQLite DB path (will be created if missing)")
    parser.add_argument("--csvdir", default="extracted_tables_2024", help="Directory with CSV files")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing entries for same filename")
    args = parser.parse_args()

    dbp = Path(args.db)
    csvdir = Path(args.csvdir)
    if not csvdir.exists():
        raise SystemExit(f"CSV directory not found: {csvdir}")

    conn = sqlite3.connect(str(dbp))
    ensure_schema(conn)

    files = sorted(csvdir.glob("*.csv"))
    n = 0
    for f in files:
        try:
            tid, hv = ingest_csv(conn, f, overwrite=args.overwrite)
            n += 1
            print(f"Imported: {f.name} -> table_id={tid} has_variety={hv}")
        except Exception as e:
            print(f"Failed to import {f.name}: {e}")
    conn.close()
    print(f"Done. {n} files processed into {dbp.resolve()}")


if __name__ == "__main__":
    main()
