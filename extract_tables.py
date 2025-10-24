# PDF table extractor with resume support (--start / --end).
# - Uses pdfplumber
# - Appends to existing CSVs in out_dir when header overlap suggests continuation
# - Processes only pages in [start, end] if provided
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
    return name[:120] if len(name) > 120 else (name or "table")


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
    # read header using csv module (fast, low mem)
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


def find_best_existing_csv(header: List[str], index, min_overlap=0.5) -> Optional[Path]:
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
    # align df to target_header (case-insensitive match), then append without header
    mapped = []
    for c in df.columns:
        matches = [tc for tc in target_header if tc.lower() == c.lower()]
        mapped.append(matches[0] if matches else c)
    df.columns = mapped
    df = df.reindex(columns=target_header, fill_value='')
    df.to_csv(path, mode='a', header=False, index=False)


def extract(pdf_path: str, out_dir: str, start_page: Optional[int], end_page: Optional[int]):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    existing_index = load_existing_csv_index(out_dir)
    print(f"Found {len(existing_index)} existing CSVs in {out_dir}", flush=True)

    last_table = None  # {name, header, last_page, csv_path}
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
                # keep last_table to allow contiguous continuation across empty pages if needed
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

                # Determine continuation logic:
                appended = False
                # 1) If title missing and last_table in this run is contiguous and header matches -> append
                if title is None and last_table and pnum == last_table["last_page"] + 1 and overlap_ratio(header, last_table["header"]) >= 0.5:
                    try:
                        safe_to_csv_append(df, last_table["csv_path"], last_table["header"])
                        last_table["last_page"] = pnum
                        appended = True
                        print(f"[page {pnum}] appended to last run table {last_table['csv_path'].name}", flush=True)
                    except Exception as exc:
                        print(f"[page {pnum}] append to last_table failed: {exc}", flush=True)

                # 2) If not appended, try matching against existing CSVs from previous run
                if not appended and title is None and existing_index:
                    candidate = find_best_existing_csv(header, existing_index, min_overlap=0.5)
                    if candidate:
                        target_hdr = [c for c in read_csv_header(candidate)]
                        target_hdr = make_unique(normalize_header(target_hdr))
                        try:
                            safe_to_csv_append(df, candidate, target_hdr)
                            # update index entry mtime
                            for it in existing_index:
                                if it["path"] == candidate:
                                    it["mtime"] = candidate.stat().st_mtime
                            last_table = {"name": candidate.stem, "header": target_hdr, "last_page": pnum, "csv_path": candidate}
                            appended = True
                            print(f"[page {pnum}] appended to existing CSV {candidate.name}", flush=True)
                        except Exception as exc:
                            print(f"[page {pnum}] append to existing CSV failed: {exc}", flush=True)

                if appended:
                    del df
                    gc.collect()
                    continue

                # New table: create file (use title if present)
                name = sanitize_filename(title) if title else f"page{pnum:03d}_table{tidx:02d}"
                csv_path = out_dir / f"{name}.csv"
                base = csv_path.stem
                k = 1
                while csv_path.exists():
                    csv_path = out_dir / f"{base}_{k}.csv"
                    k += 1
                try:
                    df.to_csv(csv_path, index=False)
                    print(f"[page {pnum}] saved new table {csv_path.name}", flush=True)
                except Exception as exc:
                    print(f"[page {pnum}] write failed ({exc}), retrying with utf-8-sig", flush=True)
                    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
                saved_files.add(str(csv_path))
                # update index for resume ability
                existing_index.append({"path": csv_path, "header": list(df.columns), "mtime": csv_path.stat().st_mtime})
                last_table = {"name": name, "header": list(df.columns), "last_page": pnum, "csv_path": csv_path}

                del df
                gc.collect()



    print(f"Done. {len(saved_files)} new tables saved/updated in {out_dir}", flush=True)

# ...existing code...


def get_starting_page_from_fname(fname):
    """
    Try to parse the starting page number from a filename like:
      "8.6_Productivity_and_expenditure_of_Potato_by_Division.csv"
      "8-9_Some_table.csv" or "12_Something.csv"
    Returns int or None.
    """
    base = os.path.basename(fname)
    # try leading number before dot/hyphen/underscore/space
    m = re.match(r'^\s*(\d+)(?:[.\-_\s]|$)', base)
    if m:
        return int(m.group(1))
    # try any pattern like "..._8_..." or "...-8_..."
    m = re.search(r'[_\-\s](\d+)(?:[_\-\s]|$)', base)
    if m:
        return int(m.group(1))
    return None

def shorten_name(original_filename):
    """
    Create a short, human-readable filename from a verbose original filename.
    Examples:
      "8.6_Productivity_and_expenditure_of_Potato_by_Division.csv" -> "Potato_Productivity_Div.csv"
      fallback -> "<Token>_Table.csv"
    """
    base = os.path.splitext(os.path.basename(original_filename))[0]
    # remove leading numbers and separators
    base = re.sub(r'^[\d\.\-_\s]+', '', base)

    # try pattern: _of_<crop>_by_
    m = re.search(r'_of_([^_]+?)_by_', base, flags=re.IGNORECASE)
    if m:
        crop = m.group(1)
    else:
        # fallback: split on separators and remove common stop words
        tokens = [t for t in re.split(r'[_\s]+', base) if t]
        stop = {'and','of','by','per','in','the','table','division','div'}
        tokens = [t for t in tokens if t.lower() not in stop]
        # prefer a token that looks like a crop (capitalized) otherwise last token
        crop = None
        for t in tokens:
            if t and t[0].isalpha() and t[0].isupper():
                crop = t
                break
        if not crop and tokens:
            crop = tokens[-1]
        if not crop:
            crop = 'Data'

    # sanitize crop token
    crop_clean = re.sub(r'[^A-Za-z0-9]', '', crop).title()

    # decide descriptor
    if re.search(r'productiv', base, flags=re.IGNORECASE):
        descriptor = 'Productivity'
    elif re.search(r'expend|expendit', base, flags=re.IGNORECASE):
        descriptor = 'Expenditure'
    elif re.search(r'area', base, flags=re.IGNORECASE):
        descriptor = 'Area'
    else:
        descriptor = 'Table'

    # prefer "Div" when division appears in original name
    suffix = 'Div' if re.search(r'division|div', base, flags=re.IGNORECASE) else ''

    parts = [crop_clean, descriptor]
    if suffix:
        parts.append(suffix)
    short = '_'.join([p for p in parts if p]) + '.csv'
    return short

# ...existing code...
# assume your extraction loop yields a list of created csv file paths -> generated_csvs
# and you have an output_dir variable where files should live.
# Replace or integrate the following saving/renaming logic into your file-export section.

# Example integration:
# for src_path in generated_csvs:
#     start_page = get_starting_page_from_fname(src_path)
#     shortname = shorten_name(src_path)
#     prefix = f"{start_page}_" if start_page is not None else ""
#     dst_name = prefix + shortname
#     dst_path = os.path.join(output_dir, dst_name)
#     # move/rename the file into final destination
#     os.makedirs(output_dir, exist_ok=True)
#     os.replace(src_path, dst_path)

# ...existing code...


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract tables from PDF to CSV with resume support")
    parser.add_argument("pdf", help="PDF file path")
    parser.add_argument("out", help="Output directory for CSVs")
    parser.add_argument("--start", type=int, default=None, help="Start page (1-based)")
    parser.add_argument("--end", type=int, default=None, help="End page (inclusive)")
    args = parser.parse_args()
    extract(args.pdf, args.out, args.start, args.end)
