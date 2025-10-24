import pdfplumber
import pandas as pd
import re
import argparse
from pathlib import Path
from typing import List, Optional, Tuple

def extract_table_title(words_above: List[dict], table_top: float, max_distance: float = 200) -> Optional[str]:
    """Extract full table title including table number."""
    relevant_words = []
    min_y = table_top - max_distance

    # Sort words by vertical position, closest to table first
    words = sorted(
        [w for w in words_above if float(w['bottom']) <= table_top and float(w['bottom']) > min_y],
        key=lambda x: -float(x['bottom'])
    )

    if not words:
        return None

    # Get lines that could form the title
    current_line = []
    current_y = None
    title_lines = []

    for word in words:
        y = round(float(word['top']), 1)

        if current_y is None:
            current_y = y

        # If on same line (within tolerance)
        if abs(y - current_y) < 5:
            current_line.append(word['text'])
        else:
            if current_line:
                line_text = ' '.join(current_line)
                title_lines.append(line_text)
                if 'Table' in line_text and len(title_lines) > 1:
                    break
            current_line = [word['text']]
            current_y = y

    if current_line:
        title_lines.append(' '.join(current_line))

    # Find the line with "Table" and combine with next line if needed
    title = None
    for i, line in enumerate(title_lines):
        if 'Table' in line:
            title = line
            if i + 1 < len(title_lines) and not any(x in line.lower() for x in ['by district', 'by division']):
                title = f"{title} {title_lines[i+1]}"
            break

    return title

def clean_table_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean extracted table data."""
    # Remove completely empty rows
    df = df.dropna(how='all')

    # Remove rows that are just numbers (likely page numbers)
    df = df[~df.iloc[:, 0].str.match(r'^\d+$', na=False)]

    # Clean cell values
    df = df.applymap(lambda x: str(x).strip() if pd.notnull(x) else '')

    return df

def generate_filename(title: str, page: int) -> str:
    """Generate appropriate filename from table title."""
    if not title:
        return f"{page}_unnamed_table.csv"

    # Extract table number if present
    table_num = ""
    m = re.search(r'Table\s+(\d+(?:\.\d+)*)', title)
    if m:
        table_num = m.group(1).replace('.', '_')

    # Extract crop name
    crop = None
    patterns = [
        r'of\s+((?:Local|Hybrid|HYV|Modern|B\.?|[A-Z]\.?)?\s*(?:Aus|Aman|Boro|Rice|Wheat|Maize|Potato|Jute|Sugarcane|Tobacco|Tea|Coffee))',
        r'of\s+([\w\s]+?)\s+(?:by|in|across)',
    ]

    for pattern in patterns:
        m = re.search(pattern, title, re.IGNORECASE)
        if m:
            crop = m.group(1).strip()
            break

    if not crop:
        return f"{page}_table_{table_num}.csv"

    # Clean crop name
    crop = re.sub(r'\s+', '_', crop)
    crop = re.sub(r'[^A-Za-z0-9_\.]', '', crop)

    # Determine type (area, production, yield, etc.)
    type_indicators = {
        'area_prod': r'area\s+and\s+production',
        'area': r'\barea\b',
        'production': r'\bproduction\b',
        'yield': r'\byield\b',
        'cost': r'\bcost\b',
        'price': r'\bprice\b',
    }

    table_type = None
    for indicator, pattern in type_indicators.items():
        if re.search(pattern, title, re.IGNORECASE):
            table_type = indicator
            break

    # Determine scope (district, division, etc.)
    scope = None
    if re.search(r'by\s+district', title, re.IGNORECASE):
        scope = 'dist'
    elif re.search(r'by\s+division', title, re.IGNORECASE):
        scope = 'div'
    elif re.search(r'by\s+upazila', title, re.IGNORECASE):
        scope = 'upz'

    # Build filename parts
    parts = [str(page), crop]
    if table_type:
        parts.append(table_type)
    if scope:
        parts.append(scope)

    return "_".join(parts) + ".csv"

def extract_tables(pdf_path: str, output_dir: str, start_page: Optional[int] = None, end_page: Optional[int] = None):
    """Extract tables from PDF with proper naming."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with pdfplumber.open(pdf_path) as pdf:
        total_pages = len(pdf.pages)
        start = max(1, start_page or 1)
        end = min(end_page or total_pages, total_pages)

        print(f"Processing pages {start} to {end}")

        current_table = None
        for page_num in range(start-1, end):
            page = pdf.pages[page_num]
            print(f"Processing page {page_num + 1}")

            # Extract words first for title detection
            try:
                words = page.extract_words()
            except Exception as e:
                print(f"Warning: Could not extract words from page {page_num + 1}: {e}")
                words = []

            # Find tables
            tables = page.find_tables()

            for table_num, table in enumerate(tables, 1):
                # Try to get title
                title = extract_table_title(words, table.bbox[1])

                # Check if this is a continuation
                is_continuation = title and bool(re.search(r'\b(Contd\.?|Continued)\b', title, re.IGNORECASE))

                if is_continuation and current_table:
                    # Append to existing table
                    try:
                        df = pd.DataFrame(table.extract())
                        df = clean_table_data(df)
                        df.to_csv(current_table['path'], mode='a', header=False, index=False)
                        print(f"Appended continuation to {current_table['path'].name}")
                    except Exception as e:
                        print(f"Error appending continuation: {e}")
                else:
                    # Extract as new table
                    try:
                        df = pd.DataFrame(table.extract())
                        df = clean_table_data(df)

                        if len(df) == 0:
                            continue

                        filename = generate_filename(title, page_num + 1)
                        output_path = output_dir / filename

                        # Ensure unique filename
                        base = output_path.stem
                        counter = 1
                        while output_path.exists():
                            output_path = output_dir / f"{base}_{counter}.csv"
                            counter += 1

                        df.to_csv(output_path, index=False)
                        print(f"Saved table to {output_path.name}")

                        current_table = {
                            'title': title,
                            'path': output_path
                        }
                    except Exception as e:
                        print(f"Error processing table: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract tables from PDF with proper naming")
    parser.add_argument("pdf", help="Path to PDF file")
    parser.add_argument("output", help="Output directory for CSV files")
    parser.add_argument("--start", type=int, help="Start page (optional)")
    parser.add_argument("--end", type=int, help="End page (optional)")

    args = parser.parse_args()
    extract_tables(args.pdf, args.output, args.start, args.end)
