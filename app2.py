# app.py
import sqlite3
import re
import json
from flask import Flask, render_template, request, redirect, url_for, g, flash, jsonify

app = Flask(__name__)
app.secret_key = 'ee01b05594e9ea2b8a9d2448fef1222951abbd044751bea9'
DATABASE = 'agri_complete.db' # Use our new database

# --- DATABASE HELPER FUNCTIONS ---

def get_db():
    db = getattr(g, '_database', None)
    if db is None:
        db = g._database = sqlite3.connect(DATABASE)
        db.text_factory = lambda b: b.decode('latin-1')
        db.row_factory = sqlite3.Row
    return db

@app.teardown_appcontext
def close_connection(exception):
    db = getattr(g, '_database', None)
    if db is not None:
        db.close()

def query_db(query, args=(), one=False):
    try:
        cur = get_db().execute(query, args)
        rv = cur.fetchall()
        cur.close()
        return (rv[0] if rv else None) if one else rv
    except sqlite3.OperationalError as e:
        print(f"Database Error: {e}")
        print(f"Query: {query}")
        print(f"Args: {args}")
        raise e # Re-raise to be caught by routes

def table_exists(table_name):
    """Checks if a table exists in the database."""
    query = "SELECT name FROM sqlite_master WHERE type='table' AND name = ?"
    result = query_db(query, [table_name], one=True)
    return result is not None

def get_district_names():
    """Fetches a sorted list of unique district names from a known table."""
    # We query a known, reliable table. 'aman_total' is a good candidate.
    # We must use the new, cleaned table name.
    districts = []
    if table_exists('Boro_total_dist'):
        districts = query_db("SELECT DISTINCT unnamed_1 FROM Boro_total_dist ORDER BY unnamed_1")

    if not districts and table_exists('Aman_total_dist'): # Fallback
        districts = query_db("SELECT DISTINCT unnamed_1 FROM Aman_total_dist ORDER BY unnamed_1")

    # Clean the list: filter out divisions, totals, and empty strings
    cleaned_districts = []
    for d in districts:
        dist_name = d['District_Division']
        if (dist_name and
            'Division' not in dist_name and
            'Divison' not in dist_name and
            'Bangladesh' not in dist_name and
            dist_name.strip()):
            cleaned_districts.append(dist_name)

    return sorted(list(set(cleaned_districts))) # Return unique, sorted list

def get_pie_chart_tables():
    """Finds all tables with names starting with 'pie_'."""
    tables = query_db("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'pie_%'")
    return [table['name'] for table in tables]

# --- CROP HIERARCHY & MASTER MAPS ---

# This is the hierarchy from your old app.py, which we will keep.
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

# MASTER MAP 1: CROP_TO_TABLE_MAP
# Maps the UI Crop Name to its primary "total" district table.
# I built this by matching CROP_HIERARCHY to your CSV screenshots.
CROP_TO_TABLE_MAP = {
    # Major Cereals
    "Aus Rice": "aus_total",
    "Aman Rice": "aman_total",
    "Boro Rice": "boro_total",
    "Wheat": "wheat_estimate",
    # Minor Cereals
    "Maize": "maize_area_prod_dist",
    "Jower (Millet)": "rabi_jower_dist", # Note: Using Rabi as default
    "Barley/Jab": "barley_area_prod_dist",
    "Cheena & Kaon": "cheena_area_prod_dist",
    "Binnidana": "binidhana_area_prod_dist",
    # Pulses
    "Lentil (Masur)": "lentil_masur_area_prod_dist",
    "Kheshari": "khesari_area_prod_dist",
    "Mashkalai": "black_gram_mashkalai_area_prod_dist",
    "Mung": "green_gram_mug_area_prod_dist",
    "Gram": "gram_area_prod_dist",
    "Motor": "pea_motor_area_prod_dist",
    "Fallon": "fallon_area_prod_dist",
    "Other Pulses": "pulses_production_dist", # Generic table
    # Oilseeds
    "Rape and Mustard": "mustard_area_prod_dist",
    "Til": "sesame_till_rabi_kharif_area_prod_dist",
    "Groundnut": "groundnut_rabi_kharif_area_prod_dist",
    "Soyabean": "soyabean_area_prod_dist",
    "Linseed": "linseed_area_prod_dist",
    "Coconut": "coconut_area_prod_dist",
    "Sunflower": "sunflower_surjamukhi_area_prod_dist",
    # Spices
    "Onion": "onion_area_prod_dist",
    "Garlic": "garlic_area_prod_dist",
    "Chillies": "rabi_chili_area_prod_dist", # Note: Using Rabi as default
    "Turmeric": "turmeric_holud_area_prod_dist",
    "Ginger": "ginger_area_prod_dist",
    "Coriander": "coriander_seed_area_prod_dist",
    "Other Spices": "spices_and_condiments_production_dist", # Generic
    # Sugar Crops
    "Sugarcane": "sugarcane_area_prod_dist",
    "Date Palm": "datepalm_juice_area_prod_dist",
    "Palmyra Palm": "palmyra_palm_juice_area_prod_dist",
    # Fibers
    "Jute": "jute_estimate",
    "Cotton": "rabi_cotton_cumilla_area_prod_dist",
    "Sunhemp": "rabi_sunhemp_area_prod_dist",
    # Narcotics
    "Tea": "tea_area_prod_dist",
    "Betelnut": "betelnut_area_prod_dist",
    "Betel Leaves": "betel_leaves_area_prod_dist",
    "Tobacco": "tobacco_jati_area_prod_dist", # Note: Using Jati as default
}


# MASTER MAP 2: CROP_VARIETY_TABLE_MAP
# Maps crops to their *specific* variety tables for the "Top Producers" page.
CROP_VARIETY_TABLE_MAP = {
    "Aus Rice": {
        "Total": "aus_total",
        "Hybrid": "aus_hybrid",
        "HYV": "aus_hyv_dist",
        "Local": "aus_local"
    },
    "Aman Rice": {
        "Total": "aman_total",
        "Hybrid": "aman_hybrid_dist",
        "HYV": "aman_hyvushil_dist",
        "Local (Transplant)": "aman_ropa_dist",
        "Local (Broadcast)": "aman_bona"
    },
    "Boro Rice": {
        "Total": "boro_total",
        "Hybrid": "boro_hybrid_dist",
        "HYV": "boro_hyv_dist", # This table *did* exist in your old DB
        "Local": "boro_local"
    },
    "Wheat": {
        "Total": "wheat_estimate" # Wheat doesn't have variety tables in this set
    },
    "Maize": {
        "Total": "maize_area_prod_dist",
        "Rabi": "rabi_maize_area_prod_dist",
        "Kharif": "kharif_maize_area_prod_dist"
    }
    # All other crops will just default to their "Total" table
}

# --- FLASK ROUTES ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/area_summary')
def area_summary():
    try:
        summary_data = query_db("SELECT * FROM Crops_indices")
    except sqlite3.OperationalError:
        flash("Could not find 'area_summary' table. Please check your database.", 'danger')
        summary_data = []

    table_headers, labels, chart_data = [], [], []
    if summary_data:
        table_headers = summary_data[0].keys()
        labels = [row['Crop'] for row in summary_data]
        chart_data = [float(str(row['Production_2023_24'] or '0').replace(',', '')) for row in summary_data]

    return render_template('area_summary.html',
                           summary_data=summary_data,
                           table_headers=table_headers,
                           labels=labels,
                           chart_data=chart_data)

@app.route('/yield_summary')
def yield_summary():
    try:
        summary_data = query_db("SELECT * FROM crops_summary")
    except sqlite3.OperationalError:
        flash("Could not find 'yield_summery' table. Please check your database.", 'danger')
        summary_data = []

    table_headers, labels, chart_data = [], [], []
    if summary_data:
        table_headers = summary_data[0].keys()
        labels = [row['Crop'] for row in summary_data]
        chart_data = [float(str(row['2023_24_Production_000_MT'] or '0').replace(',', '')) for row in summary_data]

    return render_template('yield_summary.html',
                           summary_data=summary_data,
                           table_headers=table_headers,
                           labels=labels,
                           chart_data=chart_data)

@app.route('/crop_analysis', methods=['GET', 'POST'])
def crop_analysis():
    districts = get_district_names()
    results = None
    table_headers = []
    selected_crop_form = request.form.get('crop')
    selected_district_form = request.form.get('district')

    if request.method == 'POST':
        if not selected_crop_form or not selected_district_form:
            flash("Please select both a crop and a district.", 'warning')
            return redirect(url_for('crop_analysis'))

        table_name = CROP_TO_TABLE_MAP.get(selected_crop_form)

        if not table_name:
            flash(f"Analysis for '{selected_crop_form}' is not available yet.", 'warning')
            return redirect(url_for('crop_analysis'))

        try:
            query = f"SELECT * FROM {table_name} WHERE District_Division = ?"
            results = query_db(query, [selected_district_form])

            if results:
                table_headers = results[0].keys()
            else:
                flash(f"No data found for '{selected_crop_form}' in '{selected_district_form}'.", 'info')

        except sqlite3.OperationalError:
            flash(f"Data table for '{selected_crop_form}' (tried '{table_name}') not found.", 'danger')
            results = None

    return render_template('crop_analysis.html',
                           crop_hierarchy=CROP_HIERARCHY,
                           districts=districts,
                           results=results,
                           table_headers=table_headers,
                           selected_crop=selected_crop_form,
                           selected_district=selected_district_form)

@app.route('/top_producers', methods=['GET', 'POST'])
def top_producers():
    districts = get_district_names()
    results = None
    result_type = None
    form_data = request.form.to_dict() # To repopulate form

    if request.method == 'POST':
        if 'submit_top_districts' in request.form:
            result_type = 'top_districts'
            crop = request.form.get('crop')
            variety_key = request.form.get('variety') # e.g., "Total", "Hybrid", "HYV"

            if not crop or not variety_key:
                flash("Please select a crop and a variety.", 'warning')
                return redirect(url_for('top_producers'))

            # Find the table name from our master map
            crop_varieties = CROP_VARIETY_TABLE_MAP.get(crop)
            if crop_varieties:
                table_name = crop_varieties.get(variety_key)
            else:
                # Default for all other crops
                table_name = CROP_TO_TABLE_MAP.get(crop)

            if not table_name:
                flash(f"Could not find a data table for '{crop}' - '{variety_key}'.", 'danger')
                return redirect(url_for('top_producers'))

            # Check if table exists and has the production column
            if not table_exists(table_name):
                 flash(f"Data table '{table_name}' for '{crop}' - '{variety_key}' is missing from the database.", 'danger')
                 return redirect(url_for('top_producers'))

            query = f"""
                SELECT District_Division, "2023-24_Production_MT"
                FROM {table_name}
                WHERE "2023-24_Production_MT" IS NOT NULL AND TRIM("2023-24_Production_MT") != ''
                AND District_Division != 'Bangladesh' AND District_Division NOT LIKE '%Division'
                AND District_Division NOT LIKE '%Divison'
                AND "2023-24_Production_MT" GLOB '[0-9]*'
                ORDER BY CAST(REPLACE("2023-24_Production_MT", ',', '') AS REAL) DESC
                LIMIT 10
            """

            try:
                results = query_db(query)
                if not results:
                     flash(f"No production data found for {crop} ({variety_key}).", 'info')
            except sqlite3.OperationalError:
                flash(f"Column '2023-24_Production_MT' not found in table '{table_name}'.", 'danger')

        elif 'submit_top_crops' in request.form:
            result_type = 'top_crops'
            district = request.form.get('district')

            if not district:
                flash("Please select a district.", 'warning')
                return redirect(url_for('top_producers'))

            union_queries = []
            for crop_name, table_name in CROP_TO_TABLE_MAP.items():
                if not table_exists(table_name):
                    continue

                # Check if table has the required columns
                cols_raw = query_db(f"PRAGMA table_info({table_name})")
                cols = [c['name'] for c in cols_raw]

                if "District_Division" in cols and "2023-24_Production_MT" in cols:
                    union_queries.append(f"""
                        SELECT '{crop_name}' AS Crop, "2023-24_Production_MT" AS Production
                        FROM {table_name}
                        WHERE District_Division = :district
                        AND "2023-24_Production_MT" IS NOT NULL
                        AND TRIM("2023-24_Production_MT") != ''
                        AND "2023-24_Production_MT" GLOB '[0-9]*'
                    """)

            if not union_queries:
                flash("Could not find any valid crop tables to query.", 'danger')
            else:
                full_query = " UNION ALL ".join(union_queries)
                full_query += " ORDER BY CAST(REPLACE(Production, ',', '') AS REAL) DESC"

                results = query_db(full_query, {'district': district})
                if not results:
                    flash(f"No production data found for any crop in '{district}'.", 'info')

    return render_template('top_crop_district.html',
                           crop_hierarchy=CROP_HIERARCHY,
                           districts=districts,
                           results=results,
                           result_type=result_type,
                           form_data=form_data)

@app.route('/pie_charts')
def pie_charts():
    pie_tables = get_pie_chart_tables()
    all_chart_data = []

    for table in pie_tables:
        try:
            data = query_db(f"SELECT Category, Percentage FROM {table}")
            if data:
                chart_title = table.replace('pie_', '').replace('_', ' ').title() + ' Distribution'
                labels = [row['Category'] for row in data]
                percentages = [float(row['Percentage'] or 0) for row in data]
                all_chart_data.append({
                    'title': chart_title, 'labels': labels, 'data': percentages, 'chart_id': f'chart_{table}'
                })
        except sqlite3.OperationalError:
            flash(f"Could not load data for pie chart '{table}'.", 'warning')
            continue

    return render_template('pie_charts.html', all_chart_data=all_chart_data)

# --- API ENDPOINT FOR DYNAMIC DROPDOWN ---

@app.route('/api/get_varieties_for_crop')
def get_varieties_for_crop():
    """API endpoint to get varieties for a selected crop."""
    selected_crop = request.args.get('crop')

    # Check the specific map first
    varieties_map = CROP_VARIETY_TABLE_MAP.get(selected_crop)

    if varieties_map:
        # Return the keys: ["Total", "Hybrid", "HYV", "Local"]
        return jsonify(list(varieties_map.keys()))

    # If not in the specific map, check if it's in the general map
    if selected_crop in CROP_TO_TABLE_MAP:
        # It's a crop, but has no special varieties listed
        return jsonify(["Total"])

    # If not found at all
    return jsonify([])


if __name__ == '__main__':
    app.run(debug=True)
