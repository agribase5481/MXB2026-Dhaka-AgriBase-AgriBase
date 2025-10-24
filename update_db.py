# create_db.py
import os
import sqlite3
import pandas as pd
import re

# --- CONFIGURATION ---
CSV_DIR = 'selected few'
DATABASE_NAME = 'agri_complete.db'

# This map is CRITICAL. It finds messy column names in your CSVs
# and renames them to the clean, standard name our app will use.
COLUMN_NAME_MAP = {
    # District columns
    'District/Division': 'District_Division',
    'District': 'District_Division',
    'Div/Dist': 'District_Division',
    'District_Division': 'District_Division',

    # Variety/Crop columns
    'Variety': 'Variety',
    'Crop': 'Crop',

    # Generic Area columns
    'Area_Acres': 'Area_Acres',
    'Area_Hectors': 'Area_Hectors',

    # Generic Yield columns
    'Yield_Acre_Maund': 'Yield_Acre_Maund',
    'Yield_Hector_MT': 'Yield_Hector_MT',
    'Per_Acre_Yield_Kg': 'Per_Acre_Yield_Kg',

    # Generic Production columns
    'Production_MT': 'Production_MT',
    'Production_000_MT': 'Production_000_MT',

    # Specific 2022-23 columns
    '2022-23_Area_Acres': '2022-23_Area_Acres',
    '2022-23_Area_in_acre': '2022-23_Area_Acres',
    '2022-23_Area_Hectors': '2022-23_Area_Hectors',
    '2022-23_Area_in_hector': '2022-23_Area_Hectors',
    '2022-23_Yield_Acre_Maund': '2022-23_Yield_Acre_Maund',
    '2022-23_Yield_per_acre_Maunds': '2022-23_Yield_Acre_Maund',
    '2022-23_Yield_Hector_MT': '2022-23_Yield_Hector_MT',
    '2022-23_Yield_per_hector_M_Ton': '2022-23_Yield_Hector_MT',
    '2022-23_Production_MT': '2022-23_Production_MT',
    '2022-23_Production_mton': '2022-23_Production_MT',
    '2022-23_Production_M_tons': '2022-23_Production_MT',

    # Specific 2023-24 columns
    '2023-24_Area_Acres': '2023-24_Area_Acres',
    '2023-24_Area_in_acre': '2023-24_Area_Acres',
    '2023-24_Area_Hectors': '2023-24_Area_Hectors',
    '2023-24_Area_in_hector': '2023-24_Area_Hectors',
    '2023-24_Yield_Acre_Maund': '2023-24_Yield_Acre_Maund',
    '2023-24_Yield_per_acre_Maunds': '2023-24_Yield_Acre_Maund',
    '2023-24_Yield_Hector_MT': '2023-24_Yield_Hector_MT',
    '2023-24_Yield_per_hector_M_Ton': '2023-24_Yield_Hector_MT',
    '2023-24_Production_MT': '2023-24_Production_MT',
    '2023-24_Production_mton': '2023-24_Production_MT',
    '2023-24_Production_M_tons': '2023-24_Production_MT',
}
# --- END CONFIGURATION ---

def clean_column_name(col):
    """Cleans a single column name using the map or generic rules."""
    col_stripped = str(col).strip()

    # 1. Check for a perfect match in our map
    if col_stripped in COLUMN_NAME_MAP:
        return COLUMN_NAME_MAP[col_stripped]

    # 2. If no match, apply generic cleaning
    col_cleaned = col_stripped.lower()
    col_cleaned = col_cleaned.replace(' ', '_').replace('-', '_').replace('/', '_')
    col_cleaned = re.sub(r'\(.*\)', '', col_cleaned) # Remove (text)
    col_cleaned = re.sub(r'[^a-z0-9_]+', '', col_cleaned) # Remove special chars
    col_cleaned = col_cleaned.strip('_')

    # 3. Check again if this generic version is in the map
    if col_cleaned in COLUMN_NAME_MAP:
        return COLUMN_NAME_MAP[col_cleaned]

    # 4. Fallback: return the generically cleaned name
    return col_cleaned

def generate_table_name(csv_filename):
    """Generates a clean DB table name from a CSV filename."""
    # Remove .csv extension
    table_name = os.path.splitext(csv_filename)[0]
    # Remove numeric prefix like "85_"
    table_name = re.sub(r'^\d+_', '', table_name)
    # Clean and lowercase
    table_name = table_name.lower().strip()
    table_name = table_name.replace(' ', '_').replace('-', '_')
    table_name = re.sub(r'\(.*\)', '', table_name)
    table_name = re.sub(r'[^a-z0-9_]+', '', table_name)
    return table_name.strip('_')

def import_csv_to_db(csv_file_path, table_name, conn):
    """Imports a single CSV file, cleaning column names."""
    try:
        # Read CSV as all strings, using latin-1 encoding
        df = pd.read_csv(csv_file_path, dtype=str, encoding='latin-1')

        # Clean all column names
        df.columns = [clean_column_name(col) for col in df.columns]

        # Remove duplicate columns if any
        df = df.loc[:,~df.columns.duplicated()]

        df.to_sql(table_name, conn, if_exists='replace', index=False)
        print(f"✅ Successfully imported '{csv_file_path}' into table '{table_name}'.")
    except pd.errors.EmptyDataError:
        print(f"⚠️ WARNING: '{csv_file_path}' is empty. Skipping.")
    except Exception as e:
        print(f"❌ ERROR importing '{csv_file_path}': {e}")

def main():
    if not os.path.exists(CSV_DIR):
        print(f"❌ ERROR: Directory not found: '{CSV_DIR}'")
        print("Please create it and put your 215 CSV files inside.")
        return

    # Remove old database if it exists
    if os.path.exists(DATABASE_NAME):
        os.remove(DATABASE_NAME)
        print(f"🗑️ Removed old database '{DATABASE_NAME}'.")

    conn = sqlite3.connect(DATABASE_NAME)
    print(f"✨ Created new database '{DATABASE_NAME}'.")
    print("\n--- Starting CSV Import ---")

    csv_files = [f for f in os.listdir(CSV_DIR) if f.endswith('.csv')]
    if not csv_files:
        print(f"❌ ERROR: No CSV files found in '{CSV_DIR}'.")
        conn.close()
        return

    for csv_filename in csv_files:
        csv_file_path = os.path.join(CSV_DIR, csv_filename)
        table_name = generate_table_name(csv_filename)
        import_csv_to_db(csv_file_path, table_name, conn)

    conn.close()
    print("\n--- Database creation complete! ---")
    print(f"Total tables created: {len(csv_files)}")

if __name__ == '__main__':
    main()
