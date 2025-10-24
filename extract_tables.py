# PDF table extractor tuned to produce shortened, page-prefixed CSV filenames.
# - Uses pdfplumber
# - Will append only when a table truly continues across contiguous pages (strict checks)
# - Produces filenames like "8_Potato_Productivity_Div.csv" (prefix = starting page)
# - Attempts to avoid merging/splitting errors by using stricter continuation heuristics
import argparse
import csv
import gc
import os
import re
from pathlib import Path
from typing import List, Optional, Tuple

import pdfplumber
import pandas as pd


def sanitize_filename(name: str) -> str:
    name = (name or "").strip()
    name = re.sub(r'\s+', '_', name)
    name = re.sub(r'[^0-9A-Za-z_\-\.]', '', name)
    name = re.sub(r'_+', '_', name).strip('_')
    if not name:
        name = "table"
    return name[:120] if len(name) > 120 else name


def is_mostly_text(row: List[str]) -> bool:
    non_empty = [c for c in row if str(c).strip() != ""]
    if not non_empty:
        return False
    text_like = 0
    for c in non_empty:
        s = str(c).strip()
        if re.fullmatch(r'[-+]?\d{1,3}(?:[,]\d{3})*(?:\.\d+)?|[-+]?\d+(\.\d+)?', s):
            continue
        text_like += 1
    return (text_like / len(non_empty)) >= 0.6


def row_to_list(row) -> List[str]:
    out = []
    for cell in row:
        if isinstance(cell, dict):
            out.append(cell.get("text", "").strip())
        else:
            out.append(str(cell).strip())
    return out


def title_from_words_above(table_bbox, words, max_above_px=200) -> Optional[str]:
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
    raw = " ".join(lines[best_key]).strip()
    # keep raw for detection of "Contd" but strip it for title
    m = re.search(r'(?:Table\s*\d+\s*[:\-\–]?\s*)?(.*?)(?:\bContd\.?|\bContinued\.?)?$', raw, flags=re.I)
    title = m.group(1).strip() if m else raw
    if len(title) < 3 or re.fullmatch(r'[-\s\.\d]+', title):
        return None
    return title


def normalize_header(header_row: List[str]) -> List[str]:
    return [re.sub(r'\s+', ' ', str(h).strip()) for h in header_row]


def overlap_ratio(h1: List[str], h2: List[str]) -> float:
    s1 = set([c.lower() for c in h1 if c])
    s2 = set([c.lower() for c in h2 if c])
    if not s1 or not s2:
        return 0.0
    return len(s1 & s2) / max(len(s1), len(s2))


def safe_table_extract(table) -> List[List]:
    try:
        rows = table.extract()
        return rows or []
    except Exception:
        try:
            rows = table.extract_table()
            return rows or []
        except Exception:
            return []


def make_unique(cols: List[str]) -> List[str]:
    counts = {}
    out = []
    for c in cols:
        key = c if c is not None else ""
        if key in counts:
            counts[key] += 1
            out.append(f"{key}_{counts[key]}")
        else:
            counts[key] = 0
            out.append(key)
    return out


def read_csv_header(path: Path) -> List[str]:
    try:
        with path.open('r', encoding='utf-8', errors='replace') as f:
            reader = csv.reader(f)
            header = next(reader, [])
            return [h.strip() for h in header]
    except Exception:
        return []


def load_existing_csv_index(out_dir: Path):
    index = []
    for p in sorted(out_dir.glob("*.csv")):
        hdr = read_csv_header(p)
        if hdr:
            index.append({
                "path": p,
                "header": make_unique(normalize_header(hdr)),
                "mtime": p.stat().st_mtime
            })
    return index


def find_best_existing_csv(header: List[str], index, min_overlap=0.9) -> Optional[Path]:
    # Provided for compatibility, but script will avoid appending to arbitrary old CSVs by default.
    best = None
    best_score = 0.0
    for item in index:
        score = overlap_ratio([c.lower() for c in header], [c.lower() for c in item["header"]])
        if score >= min_overlap and score > best_score:
            best_score = score
            best = item
        elif score == best_score and item["mtime"] > (best["mtime"] if best else 0):
            best = item
    return best["path"] if best else None


def safe_to_csv_append(df: pd.DataFrame, path: Path, target_header: List[str]):
    mapped = []
    for c in df.columns:
        matches = [tc for tc in target_header if tc.lower() == c.lower()]
        mapped.append(matches[0] if matches else c)
    df.columns = mapped
    df = df.reindex(columns=target_header, fill_value='')
    df.to_csv(path, mode='a', header=False, index=False)


def shorten_name_from_text(text: str) -> str:
    # try to generate a compact, descriptive name from arbitrary title/header text
    text = (text or "").strip()
    text = re.sub(r'\s+', '_', text)
    text = re.sub(r'[^0-9A-Za-z_\-]', '', text)
    text = re.sub(r'_+', '_', text).strip('_')
    if not text:
        return "Table"
    # try to extract "of_<Crop>_by" pattern
    m = re.search(r'of_([A-Za-z0-9]+?)_by', text, flags=re.I)
    if m:
        crop = m.group(1).title()
        descriptor = 'Table'
        if re.search(r'productiv', text, flags=re.I):
            descriptor = 'Productivity'
        elif re.search(r'expend', text, flags=re.I):
            descriptor = 'Expenditure'
        elif re.search(r'area', text, flags=re.I):
            descriptor = 'Area'
        suffix = 'Div' if re.search(r'division|div', text, flags=re.I) else ''
        parts = [crop, descriptor]
        if suffix:
            parts.append(suffix)
        return sanitize_filename("_".join(parts))
    # fallback: pick up first capitalized token or first token
    tokens = [t for t in re.split(r'[_\s]+', text) if t]
    stop = {'and', 'of', 'by', 'per', 'in', 'the', 'table', 'division', 'div'}
    tokens = [t for t in tokens if t.lower() not in stop]
    crop = None
    for t in tokens:
        if t and t[0].isalpha() and t[0].isupper():
            crop = t
            break
    if not crop and tokens:
        crop = tokens[0]
    crop_clean = re.sub(r'[^A-Za-z0-9]', '', (crop or 'Data')).title()
    descriptor = 'Table'
    if re.search(r'productiv', text, flags=re.I):
        descriptor = 'Productivity'
    elif re.search(r'expend', text, flags=re.I):
        descriptor = 'Expenditure'
    elif re.search(r'area', text, flags=re.I):
        descriptor = 'Area'
    suffix = 'Div' if re.search(r'division|div', text, flags=re.I) else ''
    parts = [crop_clean, descriptor]
    if suffix:
        parts.append(suffix)
    return sanitize_filename("_".join([p for p in parts if p]))


def build_short_filename(title: Optional[str], header: List[str], starting_page: int) -> Path:
    if title:
        base = shorten_name_from_text(title)
    else:
        # use first one or two header tokens to name
        tokens = [re.sub(r'[^A-Za-z0-9]', '', h).title() for h in header if h and re.sub(r'[^A-Za-z0-9]', '', h)]
        tokens = tokens[:2] if tokens else ['Table']
        base = sanitize_filename("_".join(tokens))
        # try to add descriptor if header hints at productivity/expenditure
        joined = "_".join(tokens)
        if re.search(r'productiv', joined, flags=re.I):
            base = f"{base}_Productivity"
        elif re.search(r'expend', joined, flags=re.I):
            base = f"{base}_Expenditure"
    fname = f"{starting_page}_{base}.csv"
    return Path(fname)


def extract(pdf_path: str, out_dir: str, start_page: Optional[int], end_page: Optional[int]):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    existing_index = load_existing_csv_index(out_dir)
    print(f"Found {len(existing_index)} existing CSVs in {out_dir}", flush=True)

    last_table = None  # {"start_page", "header", "last_page", "csv_path"}
    saved_files = set()

    with pdfplumber.open(pdf_path) as pdf:
        total = len(pdf.pages)
        s = start_page if start_page and start_page > 0 else 1
        e = end_page if end_page and end_page > 0 else total
        s = max(1, s)
        e = min(total, e)
        print(f"Processing pages {s}..{e} (total pages in file: {total})", flush=True)

        for pnum in range(s, e + 1):
            page = pdf.pages[pnum - 1]
            print(f"[page {pnum}]", flush=True)
            try:
                tables = page.find_tables()
            except Exception as exc:
                print(f"[page {pnum}] find_tables error: {exc}", flush=True)
                tables = []
            if not tables:
                # keep last_table to allow contiguous continuation detection on next page
                continue

            try:
                words = page.extract_words()
            except Exception:
                words = []

            for tidx, table in enumerate(tables, start=1):
                rows = safe_table_extract(table)
                if not rows:
                    continue
                norm_rows = [row_to_list(r) for r in rows]
                df = pd.DataFrame(norm_rows)

                if len(df) >= 2 and is_mostly_text(df.iloc[0].tolist()):
                    header = normalize_header(df.iloc[0].tolist())
                    header = make_unique(header)
                    df = df[1:].reset_index(drop=True)
                    df.columns = header
                else:
                    header = make_unique(normalize_header([f"col_{i}" for i in range(len(df.columns))]))
                    df.columns = header

                bbox = getattr(table, "bbox", None)
                title = None
                try:
                    title = title_from_words_above(bbox, words, max_above_px=200)
                except Exception:
                    title = None

                appended = False
                # strict continuation detection:
                if last_table:
                    # contiguous pages only
                    if pnum == last_table["last_page"] + 1:
                        header_overlap = overlap_ratio(header, last_table["header"])
                        # require strong overlap and similar column count to consider continuation
                        cols_close = abs(len(header) - len(last_table["header"])) <= 2
                        if header_overlap >= 0.8 and cols_close:
                            try:
                                safe_to_csv_append(df, last_table["csv_path"], last_table["header"])
                                last_table["last_page"] = pnum
                                appended = True
                                print(f"[page {pnum}] appended (continued) to {last_table['csv_path'].name}", flush=True)
                            except Exception as exc:
                                print(f"[page {pnum}] append to last_table failed: {exc}", flush=True)

                if appended:
                    del df
                    gc.collect()
                    continue

                # Do NOT append to arbitrary existing CSVs from previous runs to avoid accidental merges.
                # New table: create file using short name and starting page prefix
                short_path = build_short_filename(title, header, pnum)
                csv_path = out_dir / short_path.name
                base_stem = csv_path.stem
                k = 1
                # If multiple different tables on same starting page produce same short name, disambiguate
                while csv_path.exists():
                    csv_path = out_dir / f"{base_stem}_{k}.csv"
                    k += 1
                try:
                    df.to_csv(csv_path, index=False)
                    print(f"[page {pnum}] saved new table {csv_path.name}", flush=True)
                except Exception as exc:
                    print(f"[page {pnum}] write failed ({exc}), retrying with utf-8-sig", flush=True)
                    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

                saved_files.add(str(csv_path))
                # update last_table to allow proper continuation detection
                last_table = {
                    "start_page": pnum,
                    "header": list(df.columns),
                    "last_page": pnum,
                    "csv_path": csv_path
                }

                del df
                gc.collect()

    print(f"Done. {len(saved_files)} new tables saved in {out_dir}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract tables from PDF to CSV with strict continuation rules and shortened names")
    parser.add_argument("pdf", help="PDF file path")
    parser.add_argument("out", help="Output directory for CSVs")
    parser.add_argument("--start", type=int, default=None, help="Start page (1-based)")
    parser.add_argument("--end", type=int, default=None, help="End page (inclusive)")
    args = parser.parse_args()
    extract(args.pdf, args.out, args.start, args.end)
