"""
Extract tables from Agri_data_2024.pdf into separate CSVs with clear names.

Naming rules (priority):
 - prefix with starting page number (1-based)
 - crop or main subject first (e.g., Aus_Rice, Kharif_Brinjal, Crops)
 - short descriptor next (area, prod, area_prod, yield, price, exp, dist/div if by scope)
Examples:
  8_Aus_Rice_yield_dist.csv
  58_Crops_area_prod.csv

Behavior:
 - Reads table caption/title immediately above each table (captures "Table X.Y: ...")
 - Starts a new CSV whenever a caption with a table number is found
 - Only appends to previous CSV when explicit continuation appears (Contd./Continued)
 - If caption is missing, will only append when header-overlap is very high AND page is contiguous
 - Tries to avoid merging unrelated tables; preserves raw extracted cells (no forced merges/splits)
"""
from pathlib import Path
from typing import List, Optional, Tuple
import re
import argparse
import math
import csv
import os

import pdfplumber
import pandas as pd

# prioritized crop phrases (longer first). Extend as needed.
PRIORITY_CROPS = [
    "kharif brinjal", "rabi brinjal", "aus rice", "aman rice", "boro rice", "aus", "aman", "boro",
    "potato", "maize", "wheat", "jute", "sugarcane", "tobacco", "tea", "coffee",
    "cassava", "groundnut", "millet", "barley", "rapeseed", "mustard", "vegetables", "crops"
]

# descriptor patterns -> normalized descriptor
DESCRIPTORS = [
    (r'area\s+and\s+production', 'area_prod'),
    (r'area[, ]', 'area'),
    (r'\bproduction\b', 'production'),
    (r'\byield\s+rate\b|\byield\b', 'yield'),
    (r'\bproductiv', 'productivity'),
    (r'\bprice\b', 'price'),
    (r'\bexpenditur|expend', 'expenditure'),
    (r'\bestimat', 'estimate'),
]

SCOPE_PATTERNS = [
    (r'by\s+district', 'dist'),
    (r'by\s+division', 'div'),
    (r'by\s+upazila', 'upz'),
    (r'by\s+thana', 'thana'),
]


def sanitize_token(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r'\s+', '_', s)
    s = re.sub(r'[^A-Za-z0-9_]', '', s)
    return s.strip('_')


def extract_title_from_words_above(words: List[dict], table_top: float, max_distance: float = 320) -> Optional[str]:
    """
    Collect candidate words above the table area and attempt to form the table caption/title.
    Returns the title text (often includes "Table X.Y" prefix).
    """
    if not words:
        return None
    min_y = table_top - max_distance
    # select words whose bottom is above table_top but not further than min_y
    candidates = [w for w in words if float(w.get('bottom', 0)) <= table_top and float(w.get('bottom', 0)) > min_y]
    if not candidates:
        # try a larger window as a fallback
        candidates = [w for w in words if float(w.get('bottom', 0)) <= table_top and float(w.get('bottom', 0)) > table_top - (max_distance * 2)]
        if not candidates:
            return None

    # Group by approximate line (y coordinate)
    lines = {}
    for w in candidates:
        y = round(float(w.get('top', 0)), 1)
        lines.setdefault(y, []).append((float(w.get('x0', 0)), w.get('text', '').strip()))
    # sort lines by vertical position (closest to table first)
    sorted_lines = sorted(lines.items(), key=lambda kv: -kv[0])
    assembled = []
    for y, items in sorted_lines:
        items_sorted = sorted(items, key=lambda it: it[0])
        line_text = " ".join([t for _, t in items_sorted]).strip()
        if line_text:
            assembled.append((y, line_text))

    # Prefer lines containing "Table" token. Combine that line with the next one below it if it's part of title.
    for idx, (y, txt) in enumerate(assembled):
        if re.search(r'\btable\b', txt, flags=re.I) or re.search(r'^\d+(\.\d+)+', txt):
            # combine with one line immediately below (if exists and looks like continuation)
            combined = txt
            if idx + 1 < len(assembled):
                next_txt = assembled[idx + 1][1]
                # If next line doesn't start with an unrelated token like "Note", include it
                if not re.search(r'^\s*(note|source|contd|continued)\b', next_txt, flags=re.I):
                    combined = f"{combined} {next_txt}"
            return re.sub(r'\s+', ' ', combined).strip()

    # If no explicit "Table" line found, heuristically pick top-most non-empty assembled line(s) that look like a title
    if assembled:
        # pick up to two top lines joined
        top_lines = " ".join([t for _, t in assembled[:2]])
        # ensure it has alphabetic chars
        if re.search(r'[A-Za-z]', top_lines):
            return re.sub(r'\s+', ' ', top_lines).strip()

    return None


def parse_title_info(title: str) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    """
    From a title string (e.g. "Table 3.9.2 Area and Production of Kharif Brinjal by District")
    return (table_number, crop, descriptor, scope)
    """
    if not title:
        return None, None, None, None
    txt = title.strip()

    # extract table number
    table_num = None
    m = re.search(r'\btable\s+(\d+(?:\.\d+)*)', txt, flags=re.I)
    if m:
        table_num = m.group(1).replace('.', '_')

    # normalize title by removing "Table X:" prefix for easier parsing
    norm = re.sub(r'^\s*table\s*\d+(?:\.\d+)*\s*[:\-\–]?\s*', '', txt, flags=re.I)

    # Descriptor detection
    descriptor = None
    for pat, name in DESCRIPTORS:
        if re.search(pat, norm, flags=re.I):
            descriptor = name
            break

    # Scope detection
    scope = None
    for pat, name in SCOPE_PATTERNS:
        if re.search(pat, norm, flags=re.I):
            scope = name
            break

    # Crop detection - prefer explicit "of X by ..." patterns
    crop = None
    # Pattern: "of <crop> by/in/for ..."
    m = re.search(r'of\s+([A-Za-z0-9\'\-\s&()]+?)\s+(?:by|in|across|for)\b', norm, flags=re.I)
    if m:
        crop = m.group(1).strip()
    else:
        # pattern: "<Descriptor> of <crop>" or "<crop> <descriptor>"
        # try to match known priority crops (longest-first)
        low = norm.lower()
        for pc in sorted(PRIORITY_CROPS, key=lambda x: -len(x)):
            if pc in low:
                # extract the exact substring from original norm to preserve casing/parts
                idx = low.find(pc)
                crop_candidate = norm[idx: idx + len(pc)]
                crop = crop_candidate.strip()
                break

    # If still not found, try capturing first N words (for generic tables like "Area, Yield Rate and Production of Crops 2021-22 to 2023-24")
    if not crop:
        m = re.search(r'\b(crop|crops)\b', norm, flags=re.I)
        if m:
            crop = "Crops"
        else:
            # as last resort, try to capture first noun phrase up to keywords
            m = re.search(r'^(.*?)\b(?:by|in|across|for)\b', norm, flags=re.I)
            if m:
                crop = m.group(1).strip().split()[:4]  # limit length
                crop = " ".join(crop)
            else:
                crop = None

    # sanitize crop and descriptor strings
    crop = sanitize_token(crop) if crop else None
    descriptor = descriptor if descriptor else None
    scope = scope if scope else None

    return table_num, crop, descriptor, scope


def extract_table_rows_safe(table_obj) -> List[List[str]]:
    """
    Safely extract rows from a pdfplumber Table object.
    The table.extract() typically returns nested lists or dicts; normalize to list of strings.
    """
    rows = []
    # prefer table.extract() if available
    try:
        raw = table_obj.extract()
    except Exception:
        try:
            raw = table_obj.extract_table()
        except Exception:
            raw = None

    if not raw:
        return rows

    for r in raw:
        row = []
        for c in r:
            if isinstance(c, dict):
                # pdfplumber sometimes gives dict with 'text'
                row.append(str(c.get('text', '')).strip())
            else:
                row.append('' if c is None else str(c).strip())
        rows.append(row)
    return rows


def detect_header_row(rows: List[List[str]]) -> Optional[int]:
    """
    Heuristic: returns index of header row (first row which is mostly text rather than numeric).
    If none clear, return 0 as header.
    """
    for idx, row in enumerate(rows[:3]):  # check only first three rows
        non_empty = [c for c in row if str(c).strip() != ""]
        if not non_empty:
            continue
        text_like = 0
        for c in non_empty:
            s = str(c).strip()
            # number pattern
            if re.fullmatch(r'[-+]?\d{1,3}(?:[,]\d{3})*(?:\.\d+)?|[-+]?\d+(\.\d+)?', s):
                continue
            text_like += 1
        if non_empty and (text_like / len(non_empty)) >= 0.6:
            return idx
    # fallback to first row
    return 0


def header_overlap(h1: List[str], h2: List[str]) -> float:
    a = set([c.lower() for c in h1 if c])
    b = set([c.lower() for c in h2 if c])
    if not a or not b:
        return 0.0
    return len(a & b) / max(len(a), len(b))


def clean_dataframe_keep_cells(df: pd.DataFrame) -> pd.DataFrame:
    """
    Minimal cleaning: trim whitespace and replace None with empty string.
    Do NOT merge or split columns/rows here.
    """
    df = df.fillna('').astype(str)
    df = df.applymap(lambda x: x.strip())
    # drop empty rows
    df = df.loc[~(df.apply(lambda r: all([not c for c in r]), axis=1))]
    return df


def build_filename(page_start: int, crop: Optional[str], descriptor: Optional[str], scope: Optional[str], table_num: Optional[str]) -> str:
    parts = [str(page_start)]
    if crop:
        parts.append(crop)
    else:
        parts.append('table')
    if descriptor:
        parts.append(descriptor)
    if scope:
        parts.append(scope)
    # include table number if crop missing or descriptor missing to help identify
    if (not crop or not descriptor) and table_num:
        parts.append(f"t{table_num}")
    name = "_".join(parts)
    name = re.sub(r'_+', '_', name)
    # limit length
    name = name[:180]
    return f"{name}.csv"


def ensure_unique(path: Path) -> Path:
    if not path.exists():
        return path
    base = path.stem
    parent = path.parent
    for i in range(1, 1000):
        cand = parent / f"{base}_{i}.csv"
        if not cand.exists():
            return cand
    # fallback
    return parent / f"{base}_{os.getpid()}.csv"


def extract_tables(pdf_path: str, out_dir: str, start_page: Optional[int] = None, end_page: Optional[int] = None):
    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)

    with pdfplumber.open(pdf_path) as pdf:
        total = len(pdf.pages)
        s = max(1, start_page) if start_page else 1
        e = min(total, end_page) if end_page else total
        s = max(1, s)
        e = min(total, e)

        print(f"Extracting pages {s}..{e} from {pdf_path}")

        last_table_meta = None  # dict with keys: start_page, header, path, title, table_num
        saved = 0

        for pnum in range(s, e + 1):
            page = pdf.pages[pnum - 1]
            print(f"[page {pnum}] scanning...", flush=True)
            try:
                words = page.extract_words() or []
            except Exception:
                words = []

            try:
                tables = page.find_tables() or []
            except Exception:
                tables = []

            for tindex, table in enumerate(tables, start=1):
                # Attempt to extract title from words above the table bbox
                bbox = getattr(table, "bbox", None)
                table_top = bbox[1] if bbox else 0
                title = extract_title_from_words_above(words, table_top) if words else None

                # Normalize title for contd detection
                contd_marker = False
                if title and re.search(r'\b(contd\.?|continued)\b', title, flags=re.I):
                    contd_marker = True

                rows = extract_table_rows_safe(table)
                if not rows:
                    print(f"[page {pnum}] table {tindex} empty, skipping")
                    continue

                # Determine header row
                hdr_idx = detect_header_row(rows)
                header_row = rows[hdr_idx] if hdr_idx is not None and hdr_idx < len(rows) else None
                # Data rows - keep everything after header row (but keep any top caption rows if header_idx > 0)
                data_rows = rows[hdr_idx:] if hdr_idx is not None else rows

                # Build DF without implicit reshaping
                df = pd.DataFrame(data_rows)
                # Set header if header row is present (use row content as column names)
                if header_row:
                    df.columns = [str(c).strip() for c in header_row]
                    # keep rows after header
                    if len(df) > 1:
                        df = df.iloc[1:].reset_index(drop=True)
                    else:
                        df = df.iloc[0:0]  # empty
                df = clean_dataframe_keep_cells(df)
                if df.empty:
                    # sometimes header detection mis-sets, still try to save raw rows if any
                    df = pd.DataFrame(rows)
                    df = clean_dataframe_keep_cells(df)
                    if df.empty:
                        print(f"[page {pnum}] table {tindex} produced empty DataFrame after cleaning, skipping")
                        continue

                # Parse title info
                table_num, crop, descriptor, scope = parse_title_info(title) if title else (None, None, None, None)

                # If title absent and last_table exists and contiguous page and header overlap high -> treat as continuation
                is_continuation = False
                if contd_marker and last_table_meta:
                    is_continuation = True
                elif (not title) and last_table_meta and (pnum == last_table_meta.get('last_page', last_table_meta.get('start_page')) + 1):
                    # compute header overlap
                    prev_header = last_table_meta.get('header', [])
                    curr_header = list(df.columns) if df.columns is not None else []
                    if prev_header and curr_header:
                        ol = header_overlap(prev_header, curr_header)
                        if ol >= 0.9:
                            is_continuation = True

                if is_continuation and last_table_meta:
                    # Append to previous CSV (preserve prev header)
                    out_path = last_table_meta['path']
                    try:
                        df.to_csv(out_path, mode='a', header=False, index=False)
                        last_table_meta['last_page'] = pnum
                        # merge header kept same
                        print(f"[page {pnum}] appended continuation to {out_path.name}")
                        saved += 0  # no new file
                    except Exception as e:
                        print(f"[page {pnum}] failed to append continuation: {e}")
                    continue

                # Otherwise create a new CSV file for this table
                fname = build_filename(pnum, crop, descriptor, scope, table_num)
                if not fname:
                    fname = f"{pnum}_table.csv"
                out_path = out_dir_p / fname
                out_path = ensure_unique(out_path)

                # Write CSV with header
                try:
                    df.to_csv(out_path, index=False)
                    saved += 1
                    print(f"[page {pnum}] saved new table -> {out_path.name}")
                except Exception as e:
                    # try fallback encoding and quoting
                    try:
                        df.to_csv(out_path, index=False, encoding='utf-8-sig', quoting=csv.QUOTE_MINIMAL)
                        saved += 1
                        print(f"[page {pnum}] saved (fallback) -> {out_path.name}")
                    except Exception as e2:
                        print(f"[page {pnum}] failed to save table: {e2}")

                # record meta for possible continuation
                last_table_meta = {
                    'start_page': pnum,
                    'last_page': pnum,
                    'header': list(df.columns),
                    'path': out_path,
                    'title': title,
                    'table_num': table_num
                }

        print(f"Done. {saved} new table files created in {out_dir_p.resolve()}")


def main():
    parser = argparse.ArgumentParser(description="Extract tables from PDF with crop-first filenames and page prefixes")
    parser.add_argument("pdf", help="PDF file path")
    parser.add_argument("out", help="Output directory")
    parser.add_argument("--start", type=int, help="Start page (1-based)", default=None)
    parser.add_argument("--end", type=int, help="End page (inclusive)", default=None)
    args = parser.parse_args()
    extract_tables(args.pdf, args.out, args.start, args.end)


if __name__ == "__main__":
    main()
