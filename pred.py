# python
import pandas as pd
import sqlite3
from pathlib import Path
import re
import sys

SOURCE_DIRS = [Path("model/crops"), Path("model/predictions")]
DB_PATH = Path("predictions.db")

def clean_table_name(name: str) -> str:
    name = Path(name).stem
    name = re.sub(r'^\d+_', '', name)            # remove leading page numbers like "88_"
    name = re.sub(r'[^0-9a-zA-Z_]+', '_', name)  # replace invalid chars
    name = name.strip('_').lower()
    if re.match(r'^\d', name):
        name = f"t_{name}"
    return name or "table"

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [re.sub(r'\s+', '_', str(c).strip()).replace('.', '_') for c in df.columns]
    return df

def import_all_csvs(dirs, db_path):
    db_path = Path(db_path)
    conn = sqlite3.connect(str(db_path))
    imported = []
    try:
        for d in dirs:
            if not d.exists():
                print(f"Skipping missing directory: {d}", file=sys.stderr)
                continue
            for p in sorted(d.glob("*.csv")):
                table = clean_table_name(p.name)
                try:
                    # try utf-8, fallback to latin1
                    try:
                        df = pd.read_csv(p)
                    except Exception:
                        df = pd.read_csv(p, encoding="latin1")
                    if df.empty:
                        print(f"Skipping empty file: {p}", file=sys.stderr)
                        continue
                    df = normalize_columns(df)
                    df.to_sql(table, conn, if_exists="replace", index=False)
                    imported.append((p, table, len(df)))
                    print(f"Imported {p} -> {table} ({len(df)} rows)")
                except Exception as e:
                    print(f"Failed {p}: {e}", file=sys.stderr)
    finally:
        conn.commit()
        conn.close()

    print("\nSummary:")
    for src, tbl, rows in imported:
        print(f"  {src} -> {tbl} ({rows} rows)")
    print(f"Database written: {db_path} ({len(imported)} tables)")

if __name__ == "__main__":
    import_all_csvs(SOURCE_DIRS, DB_PATH)
