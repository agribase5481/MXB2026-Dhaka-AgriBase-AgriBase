import argparse
import csv
import gc
import os
import re
from pathlib import Path
from typing import List, Optional, Tuple

import pdfplumber
import pandas as pd

# Patterns for table title extraction
TABLE_TITLE_PATTERNS = [
    r'Table[:\s]+(?:\d+(?:\.\d+)*\s*)?(.+)',  # Table 3.9.2: Something
    r'(?:\d+(?:\.\d+)*\s*)?(.+?)\s*(?:by|across|in)\s+(?:District|Division)',  # Direct title with by District/Division
]

def extract_table_info(text: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Extract (crop_name, descriptor, scope) from table title text.
    Example: "Table 3.9.2: Area and Production of Kharif Brinjal by District"
    Returns: ("Kharif_Brinjal", "area_prod", "dist")
    """
    if not text:
        return None, None, None

    # Clean title without table prefix
    title = text
    for pattern in TABLE_TITLE_PATTERNS:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            title = m.group(1).strip()
            break

    # Look for crop name patterns
    crop = None
    # Pattern: "of X by" or "of X across" or "of X in"
    m = re.search(r'of\s+(.+?)\s+(?:by|across|in)\s+', title, re.IGNORECASE)
    if m:
        crop = m.group(1).strip()
    # Pattern: "X Production" or "X Area" at start
    if not crop:
        m = re.search(r'^(.+?)\s+(?:Production|Area|Yield|Price)', title, re.IGNORECASE)
        if m:
            crop = m.group(1).strip()

    # Determine descriptor
    descriptor = None
    if re.search(r'area\s+and\s+production', title, re.IGNORECASE):
        descriptor = 'area_prod'
    elif re.search(r'production', title, re.IGNORECASE):
        descriptor = 'prod'
    elif re.search(r'area', title, re.IGNORECASE):
        descriptor = 'area'
    elif re.search(r'yield', title, re.IGNORECASE):
        descriptor = 'yield'
    elif re.search(r'price', title, re.IGNORECASE):
        descriptor = 'price'

    # Determine scope
    scope = None
    if re.search(r'by\s+district', title, re.IGNORECASE):
        scope = 'dist'
    elif re.search(r'by\s+division', title, re.IGNORECASE):
        scope = 'div'
    elif re.search(r'by\s+upazila', title, re.IGNORECASE):
        scope = 'upz'

    return crop, descriptor, scope

def make_table_filename(title: str, page: int) -> str:
    """Generate filename from table title"""
    crop, descriptor, scope = extract_table_info(title)

    if not crop:
        return f"{page}_untitled_table.csv"

    # Clean crop name
    crop = re.sub(r'[^A-Za-z0-9]+', '_', crop)
    crop = re.sub(r'_+', '_', crop).strip('_')

    parts = [str(page), crop]
    if descriptor:
        parts.append(descriptor)
    if scope:
        parts.append(scope)

    return "_".join(parts) + ".csv"

def should_append_table(curr_title: str, prev_title: str) -> bool:
    """Only append if explicit continuation marker present"""
    if not curr_title or not prev_title:
        return False
    return bool(re.search(r'\bContd\.?|\bContinued\b', curr_title, re.IGNORECASE))

def title_from_words_above(table_bbox, words, max_above_px=200) -> Optional[str]:
    """Extract table title from words above the table"""
    if not table_bbox:
        return None
    x0, top, x1, bottom = table_bbox
    candidates = [
        w for w in words
        if float(w.get("bottom", 0)) < top and float(w.get("bottom", 0)) > top - max_above_px
    ]
    if not candidates:
        return None
    lines = {}
    for w in candidates:
        key = round(float(w.get("top", 0)), 1)
        lines.setdefault(key, []).append(w.get("text", ""))
    if not lines:
        return None
    best_key = sorted(lines.keys(), reverse=True)[0]
    return " ".join(lines[best_key]).strip()

def safe_table_extract(table) -> List[List]:
    """Extract table data safely"""
    try:
        rows = table.extract()
        return rows or []
    except Exception:
        try:
            rows = table.extract_table()
            return rows or []
        except Exception:
            return []

def row_to_list(row) -> List[str]:
    """Convert row to list of strings"""
    out = []
    for cell in row:
        if isinstance(cell, dict):
            out.append(cell.get("text", "").strip())
        else:
            out.append(str(cell).strip())
    return out

def is_header_row(row: List[str]) -> bool:
    """Check if row looks like a header"""
    non_empty = [c for c in row if str(c).strip()]
    if not non_empty:
        return False
    numeric_cells = 0
    for c in non_empty:
        if re.match(r'^[\d,\.]+$', str(c).strip()):
            numeric_cells += 1
    return numeric_cells / len(non_empty) < 0.5

def extract(pdf_path: str, out_dir: str, start_page: Optional[int] = None, end_page: Optional[int] = None):
    """Main extraction function"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with pdfplumber.open(pdf_path) as pdf:
        total = len(pdf.pages)
        s = start_page if start_page and start_page > 0 else 1
        e = end_page if end_page and end_page > 0 else total
        s = max(1, s)
        e = min(total, e)
        print(f"Processing pages {s}..{e} (total pages in file: {total})", flush=True)

        last_table = None
        for pnum in range(s, e + 1):
            page = pdf.pages[pnum - 1]
            print(f"[page {pnum}]", flush=True)

            try:
                tables = page.find_tables()
                words = page.extract_words()
            except Exception as exc:
                print(f"[page {pnum}] error: {exc}", flush=True)
                continue

            for tidx, table in enumerate(tables, start=1):
                # Get table title
                title = title_from_words_above(table.bbox, words)
                if not title:
                    print(f"Warning: No title found for table {tidx} on page {pnum}")
                    continue

                # Extract table data
                rows = safe_table_extract(table)
                if not rows:
                    continue

                # Convert to DataFrame
                df = pd.DataFrame([row_to_list(r) for r in rows])

                # Identify header row
                header_idx = 0
                for idx, row in df.iterrows():
                    if is_header_row(row):
                        header_idx = idx
                        break

                # Set header and remove header row from data
                if header_idx > 0:
                    df.columns = df.iloc[header_idx]
                    df = pd.concat([df.iloc[:header_idx], df.iloc[header_idx+1:]])
                df = df.reset_index(drop=True)

                # Generate filename
                filename = make_table_filename(title, pnum)
                out_path = out_dir / filename

                # Only append if explicit continuation
                if last_table and should_append_table(title, last_table["title"]):
                    try:
                        df.to_csv(last_table["path"], mode='a', header=False, index=False)
                        print(f"[page {pnum}] appended to {last_table['path'].name}", flush=True)
                    except Exception as exc:
                        print(f"[page {pnum}] append failed: {exc}", flush=True)
                else:
                    try:
                        df.to_csv(out_path, index=False)
                        print(f"[page {pnum}] saved new table {out_path.name}", flush=True)
                        last_table = {
                            "title": title,
                            "path": out_path
                        }
                    except Exception as exc:
                        print(f"[page {pnum}] save failed: {exc}", flush=True)

                del df
                gc.collect()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract tables from PDF to CSV with crop-first naming")
    parser.add_argument("pdf", help="PDF file path")
    parser.add_argument("out", help="Output directory for CSVs")
    parser.add_argument("--start", type=int, default=None, help="Start page (1-based)")
    parser.add_argument("--end", type=int, default=None, help="End page (inclusive)")
    args = parser.parse_args()
    extract(args.pdf, args.out, args.start, args.end)
