# ...existing code...
"""
PDF table extractor (pdfplumber) that:
- Prioritizes crop name first in CSV filename
- Adds a starting page number prefix (e.g. "8_aman_area.csv")
- Differentiates multiple tables for same crop via descriptor (area, district, hybrid, yield_rate, etc.)
- Attempts to avoid merging/splitting errors by strict continuation detection (only append when contiguous pages and strong header overlap)
- Saves cleaned CSVs into the specified output directory
"""
import argparse
import csv
import gc
import re
from pathlib import Path
from typing import List, Optional

import pdfplumber
import pandas as pd


def sanitize_filename(name: str) -> str:
    name = (name or "").strip()
    name = re.sub(r'\s+', '_', name)
    name = re.sub(r'[^0-9A-Za-z_\-\.]', '', name)
    name = re.sub(r'_+', '_', name).strip('_')
    name = name.lower()
    if not name:
        name = "table"
    return name[:140]


# common crop tokens prioritized for detection (lowercase)
CROP_TOKENS = [
    'rice', 'paddy', 'aus', 'aman', 'boro', 'potato', 'maize', 'corn', 'wheat', 'barley',
    'millet', 'sorghum', 'cassava', 'yam', 'groundnut', 'peanut', 'soy', 'soybean', 'tea',
    'coffee', 'sugarcane', 'banana', 'coconut', 'cotton', 'tobacco', 'onion', 'garlic',
    'tomato', 'chili', 'chilli', 'okra', 'bean', 'lentil', 'pulse', 'mustard', 'rapeseed',
    'sesame', 'chickpea', 'peas', 'vegetable', 'fruit', 'jute', 'rubber'
]


# descriptor patterns in priority order -> normalized descriptor name (lowercase)
DESCRIPTOR_PATTERNS = [
    (r'\b(hybrid|variety|varieties)\b', 'hybrid'),
    (r'\b(district|zila|thana|upazila)\b', 'district'),
    (r'\b(area|area harvested|cultivated area)\b', 'area'),
    (r'\b(yield rate|yield_per|yield|yield_rate|yield/ha)\b', 'yield_rate'),
    (r'\b(productiv|productivity)\b', 'productivity'),
    (r'\b(estimat|estimate|estimates)\b', 'estimates'),
    (r'\b(production|prod)\b', 'production'),
    (r'\b(expend|expenditure|expenditures)\b', 'expenditure'),
    (r'\b(price|prices|market)\b', 'price'),
    (r'\b(estimate|forecast)\b', 'estimate'),
    (r'\b(grade|class|category)\b', 'category'),
]

DIVISION_PATTERNS = [r'\b(division|div)\b', r'\b(by division)\b']


def extract_crop_from_text(text: Optional[str], header: Optional[List[str]] = None) -> Optional[str]:
    if text:
        cand_tokens = re.split(r'[^\w_]+', text)
        for cand in cand_tokens:
            if not cand:
                continue
            low = cand.lower()
            for ct in CROP_TOKENS:
                if ct in low:
                    return sanitize_crop_token(cand)
    if header:
        for h in header:
            if not h:
                continue
            parts = re.split(r'[^\w_]+', str(h))
            for part in parts:
                if not part:
                    continue
                low = part.lower()
                for ct in CROP_TOKENS:
                    if ct in low:
                        return sanitize_crop_token(part)
    return None


def sanitize_crop_token(token: str) -> str:
    token = (token or "").strip()
    token = re.sub(r'\s+', '_', token)
    token = re.sub(r'[^A-Za-z0-9_]', '', token)
    token = re.sub(r'_+', '_', token).strip('_')
    return token.lower()


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
    m = re.search(r'(?:Table\s*\d+\s*[:\-\–]?\s*)?(.*?)(?:\bContd\.?|\bContinued\.?)?$', raw, flags=re.I)
    title = m.group(1).strip() if m else raw
    if len(title) < 1 or re.fullmatch(r'[-\s\.\d]+', title):
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


def safe_to_csv_append(df: pd.DataFrame, path: Path, target_header: List[str]):
    mapped = []
    for c in df.columns:
        matches = [tc for tc in target_header if tc.lower() == c.lower()]
        mapped.append(matches[0] if matches else c)
    df.columns = mapped
    df = df.reindex(columns=target_header, fill_value='')
    df.to_csv(path, mode='a', header=False, index=False)


def detect_descriptor(text: Optional[str], header: Optional[List[str]] = None) -> Optional[str]:
    hay = (" ".join(header) + " " + (text or "")) if header else (text or "")
    hay = (hay or "").lower()
    for pattern, desc in DESCRIPTOR_PATTERNS:
        if re.search(pattern, hay):
            return desc
    # if none matched, try some heuristics for "district/area/productivity"
    if re.search(r'\b(division|div)\b', hay):
        return 'division'
    return None


def shorten_name_from_text(text: str, header: Optional[List[str]] = None) -> str:
    text = (text or "").strip()
    crop = extract_crop_from_text(text, header)
    descriptor = detect_descriptor(text, header) or 'table'
    # check division presence
    is_div = bool(re.search("|".join(DIVISION_PATTERNS), (" ".join(header or []) + " " + (text or "")).lower()))
    parts = []
    if crop:
        parts.append(sanitize_filename(crop))
    else:
        # if no crop, attempt to use first header token
        if header and header[0]:
            htok = re.sub(r'[^A-Za-z0-9]', '_', header[0]).strip('_').lower()
            parts.append(htok or 'table')
        else:
            parts.append('table')
    parts.append(descriptor)
    if is_div:
        parts.append('div')
    name = "_".join([p for p in parts if p])
    name = re.sub(r'_+', '_', name)
    return sanitize_filename(name)


def build_short_filename(title: Optional[str], header: List[str], starting_page: int) -> Path:
    base = shorten_name_from_text(title or "", header)
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
                # strict continuation detection for multi-page tables
                if last_table:
                    if pnum == last_table["last_page"] + 1:
                        header_overlap = overlap_ratio(header, last_table["header"])
                        cols_close = abs(len(header) - len(last_table["header"])) <= 2
                        if header_overlap >= 0.85 and cols_close:
                            try:
                                safe_to_csv_append(df, last_table["csv_path"], last_table["header"])
                                last_table["last_page"] = pnum
                                appended = True
                                print(f"[page {pnum}] appended (continued) to {last_table['csv_path'].name}", flush=True)
                            except Exception as exc:
                                print(f"[page {pnum}] append failed: {exc}", flush=True)

                if appended:
                    del df
                    gc.collect()
                    continue

                # New table - build crop-first short filename with descriptor
                short_path = build_short_filename(title, header, pnum)
                csv_path = out_dir / short_path.name
                base_stem = csv_path.stem
                k = 1
                # disambiguate if same name appears multiple times on same page or later
                while csv_path.exists():
                    csv_path = out_dir / f"{base_stem}_{k}.csv"
                    k += 1
                try:
                    df.to_csv(csv_path, index=False)
                    print(f"[page {pnum}] saved {csv_path.name}", flush=True)
                except Exception as exc:
                    print(f"[page {pnum}] write failed ({exc}), retrying with utf-8-sig", flush=True)
                    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

                saved_files.add(str(csv_path))
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
    parser = argparse.ArgumentParser(description="Extract tables from PDF to CSV with crop-first short names and page prefixes")
    parser.add_argument("pdf", help="PDF file path")
    parser.add_argument("out", help="Output directory for CSVs")
    parser.add_argument("--start", type=int, default=None, help="Start page (1-based)")
    parser.add_argument("--end", type=int, default=None, help="End page (inclusive)")
    args = parser.parse_args()
    extract(args.pdf, args.out, args.start, args.end)
# ...existing code...
