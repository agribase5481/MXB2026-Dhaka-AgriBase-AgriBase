# agribase/app.py

import sqlite3
import os
import re
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, g, flash, jsonify
# Note: You must install the google-genai library: pip install google-genai pandas

# --- CONFIGURATION ---
app = Flask(__name__)
app.secret_key = 'ee01b05594e9ea2b8a9d2448fef1222951abbd044751bea9'

# Database paths
HISTORICAL_DB = 'agri-base.db'
PREDICTIONS_DB = 'predictions.db'

# Use environment variable for API Key (Best Practice)
# FALLBACK: Use your hardcoded key from the original file if no environment variable is set
API_KEY = os.environ.get("GEMINI_API_KEY", "AIzaSyDYEx1xSi9QwRPIGL-qbAdtklFmMjj3JvQ")

# Initialize AI components globally (cached)
qa_chain = None
DB_SCHEMA_CACHE = None


# --- DATABASE HELPER FUNCTIONS (FOR FLASK VIEWS) ---

def get_db():
    """Get a database connection for the main HISTORICAL_DB from the application context."""
    db = getattr(g, '_database', None)
    if db is None:
        db = g._database = sqlite3.connect(HISTORICAL_DB)
        db.text_factory = lambda b: b.decode('latin-1')
        # sqlite3.Row makes rows accessible like dictionaries (e.g., row['Column'])
        db.row_factory = sqlite3.Row
    return db

@app.teardown_appcontext
def close_connection(exception):
    """Close the database connection at the end of the request."""
    db = getattr(g, '_database', None)
    if db is not None:
        db.close()

def query_db(query, args=(), one=False):
    """Execute a query against the main HISTORICAL_DB and return the results."""
    try:
        cur = get_db().execute(query, args)
        rv = cur.fetchall()
        cur.close()
        # Ensure we always return a list for multiple results or None/first element for one=True
        return (rv if rv else None) if one else rv
    except sqlite3.OperationalError as e:
        flash(f"Database Error: {e}. Check table names or database existence.", 'danger')
        return []

# Helper functions for crop lists and basic results cleaning
def clean_results(results):
    return results

def get_district_names():
    """Fetches a sorted list of unique district names from the database."""
    # Assuming 'aman_total_by_district' is a reliable source for districts
    districts = query_db("SELECT DISTINCT District_Division FROM aman_total_by_district ORDER BY District_Division")
    return [d['District_Division'] for d in districts]

def get_pie_chart_tables():
    """Finds all tables in the database with names starting with 'pie_'."""
    tables = query_db("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'pie_%'")
    return [table['name'] for table in tables]

# --- CROP LISTS ---
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
AVAILABLE_MAJOR_CROPS = ["Aus Rice", "Aman Rice", "Boro Rice", "Wheat"]


# --- RAG CHATBOT DATABASE & AI FUNCTIONS ---

def get_db_schema():
    """Connects to both databases and returns a single string containing the schema."""
    global DB_SCHEMA_CACHE
    if DB_SCHEMA_CACHE:
        return DB_SCHEMA_CACHE

    schema_parts = []
    dbs = {
        "HISTORICAL_DATA": HISTORICAL_DB,
        "PREDICTION_DATA": PREDICTIONS_DB
    }

    for name, db_path in dbs.items():
        if not os.path.exists(db_path):
            schema_parts.append(f"WARNING: Database file not found at {db_path}")
            continue
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT name, sql FROM sqlite_master WHERE type='table'")

            schema_parts.append(f"--- SCHEMA FOR DATABASE: {name} ({db_path}) ---")
            for table_name, sql in cursor.fetchall():
                cleaned_sql = re.sub(r'[\r\n\t]+', ' ', sql).strip()
                schema_parts.append(cleaned_sql)
            conn.close()
        except Exception as e:
            schema_parts.append(f"Error reading schema from {db_path}: {str(e)}")

    DB_SCHEMA_CACHE = "\n".join(schema_parts)
    return DB_SCHEMA_CACHE


def initialize_qa_chain():
    """Initialize Gemini chatbot and pre-fetch the database schema."""
    global qa_chain
    if qa_chain is not None:
        return True

    if not API_KEY or API_KEY == "YOUR_FALLBACK_KEY_HERE":
        print("[ERROR] GEMINI_API_KEY is not set or is using the fallback key.")
        return False

    try:
        import google.generativeai as genai
        genai.configure(api_key=API_KEY)
        qa_chain = genai.GenerativeModel('gemini-2.5-flash')
        get_db_schema()
        print("[OK] AI Chatbot initialized with gemini-2.5-flash and schema loaded.")
        return True
    except ImportError:
        print("[ERROR] 'google-generativeai' library not found. Please run: pip install google-genai")
        return False
    except Exception as e:
        print(f"[ERROR] Error initializing AI chatbot: {str(e)}")
        return False


def get_data_for_rag(sql_query):
    """
    Executes an SQL query against the appropriate database (historical or predictions).
    Returns the result as a string (DataFrame representation) or an error.
    """
    # Determine the target database based on keywords in the generated query
    db_path = PREDICTIONS_DB if "predict" in sql_query.lower() or "prediction" in sql_query.lower() else HISTORICAL_DB

    try:
        if not os.path.exists(db_path):
            return f"Error: Database file not found at {db_path}"

        conn = sqlite3.connect(db_path)
        df = pd.read_sql_query(sql_query, conn)
        conn.close()

        if df.empty:
            return "No data was returned for this specific query."

        if len(df) > 50:
            df = df.head(50)

        return df.to_string()

    except pd.io.sql.DatabaseError as e:
        return f"SQL Execution Error: The query failed. Check table and column names: {str(e)}"
    except Exception as e:
        return f"An unexpected error occurred during database access: {str(e)}"


# --- FLASK ROUTES ---

@app.before_request
def startup():
    """Initialize AI chain once"""
    global qa_chain
    if qa_chain is None:
        initialize_qa_chain()


@app.route('/')
def index():
    """Home page."""
    return render_template('index.html')


@app.route('/area_summary')
def area_summary():
    """Interactive Area Summary Report."""
    summary_data = query_db("SELECT * FROM area_summary")
    summary_data = clean_results(summary_data)

    table_headers = []
    # FIX: Get keys from the first row object if the list is not empty
    if summary_data:
        table_headers = summary_data[0].keys()
        labels = [row['Crop'] for row in summary_data]
        chart_data = [float(row['Production_2023-24'] or 0) for row in summary_data]
    else:
        labels = []
        chart_data = []

    return render_template('area_summary.html',
                            summary_data=summary_data,
                            table_headers=table_headers,
                            labels=labels,
                            chart_data=chart_data)


# /home/ubuntu/agri-base/app_54.py

# ... (lines 173-220: previous routes) ...

@app.route('/yield_summary')
def yield_summary():
    """Interactive Yield Summary Report."""
    # Assuming 'yield_summery' table exists
    summary_data = query_db("SELECT * FROM yield_summery")
    summary_data = clean_results(summary_data)

    table_headers = []
    labels = []
    chart_data = []

    if summary_data:
        # FIX 1: Get keys from the first row object. Check the list is not empty.
        table_headers = summary_data[0].keys()

        # Iterate over summary_data, accessing elements via row['Key']
        labels = [row['Crop'] for row in summary_data if 'Crop' in row.keys()]

        # FIX 2: Replace .get() with safer direct access and conversion logic.
        chart_data = []
        for row in summary_data:
            production_str = row['2023-24_Production_000_MT'] if '2023-24_Production_000_MT' in row.keys() and row['2023-24_Production_000_MT'] is not None else '0'
            try:
                # Clean up commas and convert to float
                chart_data.append(float(production_str.replace(',', '')))
            except ValueError:
                # Handle cases where data is truly non-numeric
                chart_data.append(0.0)


    return render_template('yield_summary.html',
                            summary_data=summary_data,
                            table_headers=table_headers,
                            labels=labels,
                            chart_data=chart_data)

# ... (Rest of the app remains the same, assuming prior fixes are still in place) ...
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
            flash(f"Analysis for '{selected_crop}' is not available yet.", 'warning')
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
            # FIX: Get keys from the first row object if the list is not empty
            table_headers = results[0].keys()
        else:
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
        # ... (Top Producers logic - retained from original code) ...
        if 'submit_top_districts' in request.form:
            result_type = 'top_districts'
            crop = request.form.get('crop')
            variety = request.form.get('variety')

            table_prefix_map = {"Aus Rice": "aus", "Aman Rice": "aman", "Boro Rice": "boro"}
            table_prefix = table_prefix_map.get(crop)

            if table_prefix:
                # Note: The table name here might cause an OperationalError if it doesn't exist.
                # Consider using the cleaner naming convention from the last step's utility function
                table_name = f"{table_prefix}_{variety}_by_district"
            elif crop == "Wheat":
                table_name = "wheat_estimates_district"

            query = f"""
                SELECT District_Division, "2023-24_Production_MT"
                FROM {table_name}
                WHERE "2023-24_Production_MT" IS NOT NULL AND "2023-24_Production_MT" != ''
                AND District_Division != 'Bangladesh' AND District_Division NOT LIKE '%Division'
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
                UNION ALL
                SELECT 'Aman Rice' AS Crop, CAST("2023-24_Production_MT" AS REAL) AS Production
                FROM aman_total_by_district
                WHERE District_Division = :district
                UNION ALL
                SELECT 'Boro Rice' AS Crop, CAST("2023-24_Production_MT" AS REAL) AS Production
                FROM boro_total_by_district
                WHERE District_Division = :district
                UNION ALL
                SELECT 'Wheat' AS Crop, CAST("2023-24_Production_MT" AS REAL) AS Production
                FROM wheat_estimates_district
                WHERE District_Division = :district
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
            percentages = [float(row['Percentage'] or 0) for row in data]
            all_chart_data.append({
                'title': chart_title,
                'labels': labels,
                'data': percentages,
                'chart_id': f'chart_{table}'
            })

    return render_template('pie_charts.html', all_chart_data=all_chart_data)


# --- AI CHATBOT ROUTES ---

@app.route('/ai_chatbot')
def ai_chatbot():
    """AI Chatbot page"""
    return render_template('ai_chatbot.html')


@app.route('/api/chat', methods=['POST'])
def api_chat():
    """API endpoint implementing the two-step RAG logic."""
    global qa_chain
    user_message = request.json.get('message', '').strip()

    if qa_chain is None:
        return jsonify({"success": False, "message": "AI bot not initialized. Check API key and database files."}), 503
    if not user_message:
        return jsonify({"error": "Empty message"}), 400

    db_schema = get_db_schema()

    # --- Step 1: AI generates SQL Query ---
    sql_prompt = f"""
    You are an expert SQL analyst for an agricultural database. Your goal is to write a single, optimized SQLite SQL query based on the user's question.

    DATABASE SCHEMAS:
    {db_schema}

    RULES:
    1. Only output the raw, runnable SQL query. DO NOT include any explanations, markdown (```), or comments.
    2. Use double quotes for column and table names if they contain spaces or special characters (e.g., "District_Division").
    3. If the user asks for **predictions** or **forecasts**, prioritize tables from the PREDICTION_DATA schema.
    4. Limit the result set to 20 rows using `LIMIT 20` to prevent excessive data loading.
    5. Order results by the most recent production or prediction year in descending order if applicable.

    User question: {user_message}

    SQL Query:
    """

    try:
        # Generate the SQL query
        sql_response = qa_chain.generate_content(sql_prompt)
        sql_query = sql_response.text.strip().replace('```sql', '').replace('```', '').split(';')[0].strip()

        # --- Step 2: Execute SQL Query and Retrieve Data Context ---
        data_context = get_data_for_rag(sql_query)

        # --- Step 3: AI generates Final Answer ---
        final_answer_prompt = f"""
        You are an expert agricultural consultant for AgriBase, an app dedicated to helping farmers in Bangladesh.

        Original User Question: {user_message}

        SQL Query Used to Retrieve Data: {sql_query}

        RETRIEVED DATA CONTEXT (Use this data for your answer):
        {data_context}

        INSTRUCTIONS FOR FINAL ANSWER:
        1. **Be Succinct and Highly Informative.** Directly answer the user's question.
        2. **Cite Specific Numbers and Trends.** Use exact figures from the RETRIEVED DATA CONTEXT.
        3. **Acknowledge Data Quality.** Since the data is noted as 'noisy' or from PDF extraction, use cautious language like "According to the latest data..."
        4. **Prediction/Historical Distinction.** If the data is from a prediction table, explicitly state that the figure is a **forecast** or **prediction** for the next year. If the data is historical, cite the specific year(s).
        5. **If the context is empty or contains an error, apologize and explain.**

        Final Answer:
        """

        final_response = qa_chain.generate_content(final_answer_prompt)
        answer = final_response.text

        return jsonify({"success": True, "message": answer})

    except Exception as e:
        print(f"[ERROR] Chatbot processing failed: {str(e)}")
        return jsonify({
            "success": False,
            "message": "I apologize, an internal processing error occurred. This might be due to an un-runnable SQL query or API issue. Please try rephrasing your question."
        }), 500


if __name__ == '__main__':
    initialize_qa_chain()
    app.run(debug=True)
