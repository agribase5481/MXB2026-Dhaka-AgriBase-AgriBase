import pandas as pd
import sqlite3
import os
import re
from pathlib import Path

# --- Configuration ---
# The root directory where the year folders are located
ROOT_DIR = 'merging/data'
# The name of the database file to be created
DB_NAME = 'attempt.db'
# The list of years (folder names) to process
YEARS = ['2018', '2020', '2022', '2024']

# --- Helper Function to Clean Filename ---
def create_table_name(filename, year):
    """
    Cleans the filename to create a suitable table name.

    1. Removes the file extension (.csv).
    2. Removes the initial page number (e.g., '88_').
    3. Appends the year.
    4. Replaces any remaining invalid characters (like hyphens or spaces) with underscores.
    """
    # 1. Remove the file extension
    base_name = filename.replace('.csv', '')

    # 2. Remove the initial page number (e.g., '88_') using regex
    # It looks for one or more digits followed by an underscore at the start of the string
    cleaned_name = re.sub(r'^\d+_', '', base_name)

    # 3. Append the year
    table_name = f"{cleaned_name}_{year}"

    # 4. Clean up any other potential issues for SQL table names
    table_name = table_name.replace('-', '_').replace(' ', '_').lower()

    return table_name

# --- Main Database Creation Logic ---

print(f"Starting database creation: {DB_NAME}")

# Connect to the SQLite database. It will create the file if it doesn't exist.
try:
    conn = sqlite3.connect(DB_NAME)
    print("Successfully connected to the database.")

    # Iterate over each year (folder)
    for year in YEARS:
        year_path = Path(ROOT_DIR) / year
        print(f"\n--- Processing folder: {year} ({year_path}) ---")

        # Check if the folder exists
        if not year_path.is_dir():
            print(f"⚠️ Warning: Directory not found: {year_path}. Skipping.")
            continue

        # Get all CSV files in the current year's folder
        csv_files = list(year_path.glob('*.csv'))

        if not csv_files:
            print(f"No CSV files found in {year_path}. Skipping.")
            continue

        for csv_file_path in csv_files:
            filename = csv_file_path.name

            # Create the unique table name
            table_name = create_table_name(filename, year)

            print(f"  Processing {filename} -> Table: {table_name}")

            try:
                # Read the CSV file into a pandas DataFrame
                # Assuming the first row is the header. You might need to add more options
                # like encoding='latin1' or sep=';' based on your actual data.
                df = pd.read_csv(csv_file_path)

                # Check for empty DataFrame (can happen with corrupt or empty CSVs)
                if df.empty:
                    print(f"    ⚠️ Warning: File {filename} is empty. Skipping table creation.")
                    continue

                # Load the DataFrame into the SQLite database
                # 'replace' ensures that if the script is run again, the existing table is overwritten.
                df.to_sql(
                    name=table_name,
                    con=conn,
                    if_exists='replace',
                    index=False # Do not write the DataFrame index as a column
                )

                # print(f"    ✅ Success: Table '{table_name}' created with {len(df)} rows.")

            except pd.errors.EmptyDataError:
                print(f"    ❌ Error: Could not read {filename} (Empty data or parsing issue).")
            except Exception as e:
                print(f"    ❌ An error occurred while processing {filename}: {e}")

    # Commit all changes and close the connection
    conn.commit()
    conn.close()
    print(f"\n✅ Database '{DB_NAME}' created and populated successfully.")
    print("Total tables created: (Check your database file).")

except Exception as e:
    print(f"\n❌ A critical error occurred: {e}")
