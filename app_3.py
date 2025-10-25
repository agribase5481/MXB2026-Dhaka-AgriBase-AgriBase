# agribase/app.py

import sqlite3
from flask import Flask, render_template, request, redirect, url_for, g, flash


app = Flask(__name__)
app.secret_key = 'ee01b05594e9ea2b8a9d2448fef1222951abbd044751bea9'  # Needed for flash message
DATABASE = 'attempt.db'

# --- DATABASE HELPER FUNCTIONS ---

def get_db():
    """Get a database connection from the application context."""
    db = getattr(g, '_database', None)
    if db is None:
        db = g._database = sqlite3.connect(DATABASE)



        # FIX 2: Handle text encoding errors like in 'Cox's Bazar'.
        # This tells sqlite3 to use the 'latin-1' encoding, which prevents decoding crashes.
        db.text_factory = lambda b: b.decode('latin-1')

        # Return rows as dictionaries
        db.row_factory = sqlite3.Row
    return db

@app.teardown_appcontext
def close_connection(exception):
    """Close the database connection at the end of the request."""
    db = getattr(g, '_database', None)
    if db is not None:
        db.close()

def query_db(query, args=(), one=False):
    """Execute a query and return the results."""
    cur = get_db().execute(query, args)
    rv = cur.fetchall()
    cur.close()
    return (rv[0] if rv else None) if one else rv

def get_district_names():
    """Fetches a sorted list of unique district names from the database."""
    districts = query_db('SELECT DISTINCT "Unnamed: 1" AS District_Division FROM aman_total_dist_2024 ORDER BY District_Division')
    return [d['District_Division'] for d in districts]

def get_pie_chart_tables():
    """Finds all tables in the database with names starting with 'pie_'."""
    tables = query_db("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'pie_%'")
    return [table['name'] for table in tables]

def clean_results(results):
    cleaned = results
    return cleaned



# New helpers
def table_exists(name: str) -> bool:
    cur = get_db().execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (name,))
    r = cur.fetchone()
    cur.close()
    return bool(r)

def list_tables() -> List[str]:
    return [r['name'] for r in query_db("SELECT name FROM sqlite_master WHERE type='table'")]

def get_table_columns(table: str) -> List[str]:
    cur = get_db().execute(f"PRAGMA table_info('{table}')")
    rows = cur.fetchall()
    cur.close()
    return [r[1] for r in rows] if rows else []

def find_table_for_crop(crop: str) -> Optional[str]:
    """Try common table name patterns, otherwise fall back to partial match."""
    slug = re.sub(r'[^0-9a-z]+', '_', crop.lower()).strip('_')
    candidates = [
        f"{slug}_total_by_district",
        f"{slug}_by_district",
        f"{slug}_total",
        f"{slug}_estimates_district",
        f"{slug}_estimates",
        slug
    ]
    for c in candidates:
        if table_exists(c):
            return c
    # fallback: any table that contains the slug fragment
    for t in list_tables():
        if slug in t:
            return t
    return None

def find_column(table: str, candidates: List[str]) -> Optional[str]:
    cols = get_table_columns(table)
    for cand in candidates:
        for c in cols:
            if cand.lower() == c.lower() or cand.lower() in c.lower():
                return c
    return None

def find_district_column(table: str) -> Optional[str]:
    return find_column(table, ['District_Division', 'District', 'Unnamed: 1', 'District_Div'])

def find_production_column(table: str) -> Optional[str]:
    # heuristics to find production column (2023-24 or similar)
    col = find_column(table, ['2023-24', '2023_24', '2023', 'Production_2023-24', '2023-24_Production_MT', '2023_24_Production_MT', 'Production_MT', 'Production'])
    if col:
        return col
    # fallback: pick first column that looks numeric by sampling one row
    try:
        cur = get_db().execute(f'SELECT * FROM "{table}" LIMIT 1')
        row = cur.fetchone()
        cur.close()
        if row:
            for k in row.keys():
                v = row[k]
                try:
                    float(str(v).replace(',', ''))
                    return k
                except Exception:
                    continue
    except Exception:
        pass
    return None




# --- CROP LISTS ---
# Full list of crops for the dropdowns, based on your screenshots
CROP_HIERARCHY = {
    "Major Cereals": ["Aus Rice", "Aman Rice", "Boro Rice", "Wheat"],
    "Minor Cereals": ["Maize", "Jower (Millet)", "Barley/Jab", "Cheena & Kaon", "Binnidana"],
    "Pulses": ["Lentil (Masur)", "Kheshari", "Mashkalai", "Mung", "Gram", "Motor", "Fallon", "Other Pulses"],
    "Oilseeds": ["Rape and Mustard", "Til", "Groundnut", "Soyabean", "Linseed", "Coconut", "Sunflower"],
    "Spices": ["Onion", "Garlic", "Chillies", "Turmeric", "Ginger", "Coriander", "Other Spices"],
    "Sugar Crops": ["Sugarcane", "Date Palm", "Palmyra Palm"],
    "Fibers": ["Jute", "Cotton", "Sunhemp"],
    "Narcotics": ["Tea", "Betelnut", "Betel Leaves", "Tobacco"],
}

# Crops for which we currently have data tables
AVAILABLE_MAJOR_CROPS = ["Aus Rice", "Aman Rice", "Boro Rice", "Wheat", "Maize", "Jower (Millet)", "Barley/Jab", "Cheena & Kaon", "Binnidana", "Lentil (Masur)",
                         "Kheshari", "Mashkalai", "Mung", "Gram", "Motor", "Fallon", "Other Pulses", "Rape and Mustard", "Til", "Groundnut", "Soyabean", "Linseed", "Coconut", "Sunflower",
                         "Onion", "Garlic", "Chillies", "Turmeric", "Ginger", "Coriander", "Other Spices", "Sugarcane", "Date Palm", "Palmyra Palm", "Jute", "Cotton", "Sunhemp",
                         "Tea", "Betelnut", "Betel Leaves", "Tobacco"
                         ]

# --- FLASK ROUTES ---

@app.route('/')
def index():
    """Home page."""
    return render_template('index.html')

# A common pattern for both routes
@app.route('/area_summary')
def area_summary():
    """Interactive Area Summary Report."""
    summary_data = query_db("SELECT * FROM area_summary")
    summary_data = clean_results(summary_data)

    # Initialize headers to an empty list
    table_headers = []

    # ONLY get headers and chart data if there's actual data
    if summary_data:
        table_headers = summary_data[0].keys()
        labels = [row['Crop'] for row in summary_data]
        chart_data = [float(row['Production_2023-24'] or 0) for row in summary_data]
    else:
        # If no data, ensure these are empty too, to prevent errors in the template
        labels = []
        chart_data = []

    return render_template('area_summary.html',
                           summary_data=summary_data,
                           table_headers=table_headers, # Pass the (possibly empty) headers
                           labels=labels,
                           chart_data=chart_data)

@app.route('/yield_summary')
def yield_summary():
    """Interactive Yield Summary Report."""
    summary_data = query_db("SELECT * FROM yield_summary")

    # ✅ Apply cleaning
    summary_data = clean_results(summary_data)

    # Initialize headers, labels, and chart data to empty lists
    table_headers = []
    labels = []
    chart_data = []

    # ✅ Apply safe logic: ONLY process data if results exist
    if summary_data:
        # 1. Get headers safely
        table_headers = summary_data[0].keys()

        # 2. Extract labels
        labels = [row['Crop'] for row in summary_data]

        # 3. Extract and safely convert chart data
        # Using the same safe conversion method here for consistency
        chart_data = [
            float((row['2023-24_Production_MT'] or '0').replace(',', ''))
            for row in summary_data
        ]

    return render_template('yield_summary.html',
                           summary_data=summary_data,
                           table_headers=table_headers,
                           labels=labels,
                           chart_data=chart_data)

@app.route('/crop_analysis', methods=['GET', 'POST'])
def crop_analysis():
    """Crop analysis page to search data by crop and district."""
    districts = get_district_names()
    results = None
    table_headers = []

    if request.method == 'POST':
        selected_crop = request.form.get('crop')
        selected_district = request.form.get('district')

        if selected_crop not in AVAILABLE_MAJOR_CROPS:
            flash(f"Analysis for '{selected_crop}' is not available yet. Please select one of the major cereals.", 'warning')
            return redirect(url_for('crop_analysis'))

        table_prefix_map = {
            "Aus Rice": "aus",
            "Aman Rice": "aman",
            "Boro Rice": "boro",
            "Wheat": "wheat"
        }
        table_prefix = table_prefix_map.get(selected_crop)

        table_name = "wheat_estimates_district" if table_prefix == "wheat" else f"{table_prefix}_total_by_district"

        query = f"SELECT * FROM {table_name} WHERE District_Division = ?"
        results = query_db(query, [selected_district])

        if results:
            results = clean_results(results)
            # Only set headers if results exist
            table_headers = results[0].keys()
        else:
            # If no results, explicitly ensure headers are an empty list
            table_headers = []

    return render_template('crop_analysis.html',
                           crop_hierarchy=CROP_HIERARCHY,
                           districts=districts,
                           results=results,
                           table_headers=table_headers)

@app.route('/top_producers', methods=['GET', 'POST'])
def top_producers():
    """Page to find top producing districts for a crop, or top crops for a district."""
    districts = get_district_names()
    results = None
    result_type = None

    if request.method == 'POST':
        if 'submit_top_districts' in request.form:
            result_type = 'top_districts'
            crop = request.form.get('crop')
            variety = request.form.get('variety')

            table_prefix_map = {"Aus Rice": "aus", "Aman Rice": "aman", "Boro Rice": "boro"}
            table_prefix = table_prefix_map.get(crop)

            if table_prefix:
                table_name = f"{table_prefix}_{variety}_by_district"
            elif crop == "Wheat":
                    table_name = "wheat_estimates_district"

            query = f"""
                SELECT District_Division, "2023-24_Production_MT"
                FROM {table_name}
                WHERE "2023-24_Production_MT" IS NOT NULL AND "2023-24_Production_MT" != ''
                AND District_Division != 'Bangladesh' AND District_Division NOT LIKE '%Division'
                AND District_Division NOT LIKE '%Divison'
                ORDER BY CAST("2023-24_Production_MT" AS REAL) DESC
                LIMIT 10
            """
            results = query_db(query)
            results = clean_results(results)


        elif 'submit_top_crops' in request.form:
            result_type = 'top_crops'
            district = request.form.get('district')

            query = """
                SELECT 'Aus Rice' AS Crop, CAST("2023-24_Production_MT" AS REAL) AS Production
            FROM aus_total_by_district
            WHERE District_Division = :district
            AND "2023-24_Production_MT" IS NOT NULL
            AND TRIM("2023-24_Production_MT") != ''
            AND "2023-24_Production_MT" GLOB '[0-9]*'
            AND "2023-24_Production_MT" GLOB '*[0-9]*'
            AND "2023-24_Production_MT" NOT GLOB '*[^0-9.]*'
            UNION ALL
            SELECT 'Aman Rice' AS Crop, CAST("2023-24_Production_MT" AS REAL) AS Production
            FROM aman_total_by_district
            WHERE District_Division = :district
            AND "2023-24_Production_MT" IS NOT NULL
            AND TRIM("2023-24_Production_MT") != ''
            AND "2023-24_Production_MT" GLOB '[0-9]*'
            AND "2023-24_Production_MT" GLOB '*[0-9]*'
            AND "2023-24_Production_MT" NOT GLOB '*[^0-9.]*'
            UNION ALL
            SELECT 'Boro Rice' AS Crop, CAST("2023-24_Production_MT" AS REAL) AS Production
            FROM boro_total_by_district
            WHERE District_Division = :district
            AND "2023-24_Production_MT" IS NOT NULL
            AND TRIM("2023-24_Production_MT") != ''
            AND "2023-24_Production_MT" GLOB '[0-9]*'
            AND "2023-24_Production_MT" GLOB '*[0-9]*'
            AND "2023-24_Production_MT" NOT GLOB '*[^0-9.]*'
            UNION ALL
            SELECT 'Wheat' AS Crop, CAST("2023-24_Production_MT" AS REAL) AS Production
            FROM wheat_estimates_district
            WHERE District_Division = :district
            AND "2023-24_Production_MT" IS NOT NULL
            AND TRIM("2023-24_Production_MT") != ''
            AND "2023-24_Production_MT" GLOB '[0-9]*'
            AND "2023-24_Production_MT" GLOB '*[0-9]*'
            AND "2023-24_Production_MT" NOT GLOB '*[^0-9.]*'
            ORDER BY Production DESC
            """
            results = query_db(query, {'district': district})
            results = clean_results(results)

    return render_template('top_crop_district.html',
                           available_crops=AVAILABLE_MAJOR_CROPS,
                           districts=districts,
                           results=results,
                           result_type=result_type)

@app.route('/pie_charts')
def pie_charts():
    """Generates and displays pie charts for all 'pie_' tables."""
    pie_tables = get_pie_chart_tables()
    all_chart_data = []

    for table in pie_tables:
        data = query_db(f"SELECT Category, Percentage FROM {table}")
        data = clean_results(data)
        if data:
            chart_title = table.replace('pie_', '').replace('_', ' ').title() + ' Distribution'
            labels = [row['Category'] for row in data]
            percentages = [(row['Percentage'] or 0) for row in data]
            all_chart_data.append({
                'title': chart_title,
                'labels': labels,
                'data': percentages,
                'chart_id': f'chart_{table}'
            })

    return render_template('pie_charts.html', all_chart_data=all_chart_data)

if __name__ == '__main__':
    app.run(debug=True)
