# agribase/app.py

import sqlite3
from flask import Flask, render_template, request, redirect, url_for, g, flash, jsonify
import os
import pandas as pd


app = Flask(__name__)
app.secret_key = 'ee01b05594e9ea2b8a9d2448fef1222951abbd044751bea9'
DATABASE = 'agri-base.db'
AI_DATABASE = 'predictions.db'  # Separate DB for AI bot
api_key = "AIzaSyDYEx1xSi9QwRPIGL-qbAdtklFmMjj3JvQ"

# Initialize LangChain components globally (cached)
qa_chain = None


# --- DATABASE HELPER FUNCTIONS ---

def get_db():
    """Get a database connection from the application context."""
    db = getattr(g, '_database', None)
    if db is None:
        db = g._database = sqlite3.connect(DATABASE)
        db.text_factory = lambda b: b.decode('latin-1')
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
    return (rv if rv else None) if one else rv


def get_district_names():
    """Fetches a sorted list of unique district names from the database."""
    districts = query_db("SELECT DISTINCT District_Division FROM aman_total_by_district ORDER BY District_Division")
    return [d['District_Division'] for d in districts]


def get_pie_chart_tables():
    """Finds all tables in the database with names starting with 'pie_'."""
    tables = query_db("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'pie_%'")
    return [table['name'] for table in tables]


def clean_results(results):
    return results


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


# --- AI CHATBOT FUNCTIONS ---
#api_key = "AIzaSyDYEx1xSi9QwRPIGL-qbAdtklFmMjj3JvQ"
def initialize_qa_chain():
    """Initialize simple Gemini chatbot with gemini-2.5-flash"""
    global qa_chain

    try:

        if not api_key:
            print("[WARNING] GEMINI_API_KEY not set")
            return False

        import google.generativeai as genai

        genai.configure(api_key=api_key)

        # Use the stable gemini-2.5-flash model
        qa_chain = genai.GenerativeModel('gemini-2.5-flash')

        print("[OK] AI Chatbot initialized with gemini-2.5-flash")
        return True

    except Exception as e:
        print(f"[ERROR] Error initializing AI chatbot: {str(e)}")
        return False


# Initialize on app startup
@app.before_request
def startup():
    """Initialize AI chain once"""
    global qa_chain
    if qa_chain is None:
        initialize_qa_chain()


# --- FLASK ROUTES ---

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
    if summary_data:
        table_headers = summary_data.keys()
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


@app.route('/yield_summary')
def yield_summary():
    """Interactive Yield Summary Report."""
    summary_data = query_db("SELECT * FROM yield_summery")
    summary_data = clean_results(summary_data)

    table_headers = []
    labels = []
    chart_data = []

    if summary_data:
        table_headers = summary_data.keys()
        labels = [row['Crop'] for row in summary_data]
        chart_data = [
            float((row['2023-24_Production_000_MT'] or '0').replace(',', ''))
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
            table_headers = results.keys()
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
    """API endpoint with smart data loading"""
    global qa_chain

    try:
        data = request.json
        user_message = data.get('message', '').strip()

        if not user_message:
            return jsonify({"error": "Empty message"}), 400

        if qa_chain is None:
            return jsonify({
                "success": False,
                "message": "AI bot not initialized"
            }), 503

        import sqlite3
        conn = sqlite3.connect(AI_DATABASE)
        cursor = conn.cursor()

        # Get table names
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' and name like '%_dist%' ")
        all_tables = [t[0] for t in cursor.fetchall()]

        # Build context - check if user is asking about predictions or historical data
        data_context = ""

        if 'predict' in user_message.lower() or 'forecast' in user_message.lower() or 'next' in user_message.lower():
            # Include prediction tables
            data_context = "NEXT YEAR PRODUCTION PREDICTIONS:\n\n"
            for table_name in all_tables:
                if '_predictions' in table_name:
                    try:
                        df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
                        crop_name = table_name.replace('_predictions', '')
                        data_context += f"\nCrop: {crop_name}\n"
                        data_context += df.to_string()
                        data_context += "\n"
                    except:
                        pass
        else:
            # Include historical data for the crops mentioned in question
            data_context = "HISTORICAL CROP DATA:\n\n"
            for table_name in all_tables:
                if '_predictions' not in table_name:
                    try:
                        df = pd.read_sql_query(f"SELECT * FROM {table_name} LIMIT 10", conn)
                        if not df.empty:
                            data_context += f"\nCrop: {table_name}\n"
                            data_context += df.to_string()
                            data_context += "\n"
                    except:
                        pass

        conn.close()

        # Enhanced prompt
        prompt = f"""You are an expert agricultural analyst with access to Bangladesh crop production data.

{data_context}

User question: {user_message}

Please provide an accurate, data-driven response:
1. Use specific numbers from the data provided
2. Reference district names and years
3. For predictions, clearly indicate these are forecasts for next year
4. For historical data, reference the actual years in the dataset
5. Be concise but informative"""

        # Get response
        response = qa_chain.generate_content(prompt)
        answer = response.text if response.text else "I couldn't generate a response."

        return jsonify({
            "success": True,
            "message": answer
        })

    except Exception as e:
        print(f"[ERROR] Error in /api/chat: {str(e)}")
        return jsonify({
            "success": False,
            "message": f"Sorry, I encountered an error: {str(e)}"
        }), 500




@app.route('/api/chat/health', methods=['GET'])
def chat_health():
    """Check if chatbot is ready"""
    global qa_chain
    return jsonify({
        "ready": qa_chain is not None,
        "api_key_set": bool(api_key),
        "ai_db_exists": os.path.exists(AI_DATABASE)
    })


if __name__ == '__main__':
    app.run(debug=True)
