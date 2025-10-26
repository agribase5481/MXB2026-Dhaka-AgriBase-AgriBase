# ...existing code...
import sqlite3
import os
import re
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, g, flash, jsonify
import difflib
import math
# Note: You must install the google-genai library: pip install google-genai pandas

# --- CONFIGURATION ---
app = Flask(__name__)
app.secret_key = 'ee01b05594e9ea2b8a9d2448fef1222951abbd044751bea9'

# Database paths
HISTORICAL_DB = 'agri-base.db'
PREDICTIONS_DB = 'predictions.db'
ATTEMPT_DB = 'attempt.db'                   # <-- new DB the user requested

# Use environment variable for API Key (Best Practice)
API_KEY = os.environ.get("GEMINI_API_KEY", None)

# Initialize AI components globally (cached)
qa_chain = None
DB_SCHEMA_CACHE = None
RAG_DOCS_CACHE = None    # list of dicts: {'id','db','table','schema','sample_text','text'}
# ...existing code...

def get_db():
    """Get a database connection for the main HISTORICAL_DB from the application context."""
    db = getattr(g, '_database', None)
    if db is None:
        db = g._database = sqlite3.connect(HISTORICAL_DB)
        db.text_factory = lambda b: b.decode('latin-1')
        db.row_factory = sqlite3.Row
    return db

# ...existing code...

def get_db_schema():
    """Connects to all databases (historical, predictions, attempt) and returns schema + cached sample rows.
       Also populates RAG_DOCS_CACHE used for lightweight retrieval."""
    global DB_SCHEMA_CACHE, RAG_DOCS_CACHE
    if DB_SCHEMA_CACHE and RAG_DOCS_CACHE:
        return DB_SCHEMA_CACHE

    schema_parts = []
    docs = []
    dbs = {
        "HISTORICAL_DATA": HISTORICAL_DB,
        "PREDICTION_DATA": PREDICTIONS_DB,
        "ATTEMPT_DATA": ATTEMPT_DB
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
                if not sql:
                    continue
                cleaned_sql = re.sub(r'[\r\n\t]+', ' ', sql).strip()
                schema_parts.append(cleaned_sql)

                # Try to sample up to 5 rows for retrieval context
                try:
                    c2 = conn.cursor()
                    c2.execute(f'SELECT * FROM "{table_name}" LIMIT 5')
                    rows = c2.fetchall()
                    colnames = [d[0] for d in c2.description] if c2.description else []
                    sample_lines = []
                    for r in rows:
                        # r may be a tuple
                        vals = [str(r[i]) if r[i] is not None else '' for i in range(len(colnames))]
                        sample_lines.append(", ".join(f"{col}:{vals[i]}" for i, col in enumerate(colnames)))
                    sample_text = " | ".join(sample_lines) if sample_lines else ""
                    doc_text = f"TABLE: {table_name}\nSCHEMA: {cleaned_sql}\nSAMPLE: {sample_text}"
                    docs.append({
                        'id': f"{name}:{table_name}",
                        'db': name,
                        'path': db_path,
                        'table': table_name,
                        'schema': cleaned_sql,
                        'sample_text': sample_text,
                        'text': doc_text
                    })
                except Exception:
                    # skip sampling on error, but continue
                    pass

            conn.close()
        except Exception as e:
            schema_parts.append(f"Error reading schema from {db_path}: {str(e)}")

    DB_SCHEMA_CACHE = "\n".join(schema_parts)
    RAG_DOCS_CACHE = docs
    return DB_SCHEMA_CACHE

def _token_overlap_score(query, text):
    """Simple token overlap scoring for lightweight retrieval."""
    q_tokens = set(re.findall(r'\w+', query.lower()))
    t_tokens = set(re.findall(r'\w+', text.lower()))
    if not q_tokens or not t_tokens:
        return 0.0
    overlap = len(q_tokens & t_tokens)
    # normalize by log sizes to favor dense matches
    return overlap / (math.log(len(q_tokens) + 1) + math.log(len(t_tokens) + 1))

def retrieve_relevant_docs(query, top_k=3):
    """Return top-k most relevant doc dicts from RAG_DOCS_CACHE using token overlap.
       Attempt to use embeddings from genai if available (optional fallback)."""
    global RAG_DOCS_CACHE
    if RAG_DOCS_CACHE is None:
        get_db_schema()  # builds cache

    docs = RAG_DOCS_CACHE or []
    if not docs:
        return []

    # Try embeddings-based retrieval if genai supports embeddings and API key present
    try:
        import google.generativeai as genai
        genai.configure(api_key=API_KEY) if API_KEY else None
        if hasattr(genai, "Embeddings") or hasattr(genai, "embeddings"):  # tentative check
            # This is a best-effort; the exact API may differ by release. If it fails, fallback.
            try:
                # create embeddings for query and docs (best-effort)
                model_name = "embed_text_1" if hasattr(genai, "Embeddings") else "textembedding-gecko"
                # This block may raise if API/SDK differs; we ignore and fallback.
                q_emb = genai.embeddings.create(model=model_name, input=query).embeddings[0].embedding
                scores = []
                for d in docs:
                    d_emb = genai.embeddings.create(model=model_name, input=d['text']).embeddings[0].embedding
                    # cosine similarity
                    dot = sum(a * b for a, b in zip(q_emb, d_emb))
                    norm_q = math.sqrt(sum(a * a for a in q_emb))
                    norm_d = math.sqrt(sum(b * b for b in d_emb))
                    sim = dot / (norm_q * norm_d + 1e-12)
                    scores.append((sim, d))
                scores.sort(key=lambda x: x[0], reverse=True)
                return [d for _, d in scores[:top_k]]
            except Exception:
                pass
    except Exception:
        pass

    # Fallback: token overlap
    scored = []
    for d in docs:
        s = _token_overlap_score(query, d['text'])
        scored.append((s, d))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [d for s, d in scored[:top_k] if s > 0] or [d for _, d in scored[:top_k]]

def initialize_qa_chain():
    """Initialize Gemini chatbot and pre-fetch the database schema."""
    global qa_chain
    if qa_chain is not None:
        return True

    if not API_KEY:
        print("[WARN] GEMINI_API_KEY is not set. AI features will fail without a valid key.")
        return False

    try:
        import google.generativeai as genai
        genai.configure(api_key=API_KEY)
        # Use the generative model for content generation
        qa_chain = genai.GenerativeModel('gemini-2.5-flash')
        # prime the schema and doc cache
        get_db_schema()
        print("[OK] AI Chatbot initialized with gemini-2.5-flash and schema loaded.")
        return True
    except ImportError:
        print("[ERROR] 'google-generativeai' library not found. Please run: pip install google-genai")
        return False
    except Exception as e:
        print(f"[ERROR] Error initializing AI chatbot: {str(e)}")
        return False

def get_data_for_rag(sql_query, prefer_db=None):
    """
    Executes an SQL query against the appropriate database (historical, predictions, or attempt).
    Returns the result as a string (DataFrame representation) or an error.
    """
    # Basic safety: only allow SELECT queries
    if not re.match(r'^\s*SELECT\b', sql_query, re.IGNORECASE):
        return "Error: Only SELECT queries are allowed in this interface."

    # Heuristic selection: if a table name from RAG_DOCS_CACHE matches, use its db path
    target_db = None
    if prefer_db:
        target_db = prefer_db
    else:
        # search for any table mentioned in query
        if RAG_DOCS_CACHE is None:
            get_db_schema()
        for d in (RAG_DOCS_CACHE or []):
            if re.search(rf'\b{re.escape(d["table"])}\b', sql_query, re.IGNORECASE):
                target_db = d['path']
                break

    # fallback: use predictions if query mentions predict/forecast/forecasted, else historical
    if not target_db:
        if re.search(r'predict|forecast|prediction|forecasted|expected', sql_query, re.IGNORECASE):
            target_db = PREDICTIONS_DB if os.path.exists(PREDICTIONS_DB) else HISTORICAL_DB
        else:
            # default to historical
            target_db = HISTORICAL_DB

    try:
        if not os.path.exists(target_db):
            return f"Error: Database file not found at {target_db}"

        conn = sqlite3.connect(target_db)
        df = pd.read_sql_query(sql_query, conn)
        conn.close()

        if df.empty:
            return "No data was returned for this specific query."

        if len(df) > 100:
            df = df.head(100)

        # return concise string and also JSON-ready representation if needed
        return df.to_string(index=False)
    except pd.io.sql.DatabaseError as e:
        return f"SQL Execution Error: The query failed. Check table and column names: {str(e)}"
    except Exception as e:
        return f"An unexpected error occurred during database access: {str(e)}"

# ...existing routes and functions remain mostly unchanged until api_chat...
# Replace the /api/chat route implementation with the improved RAG flow:

@app.route('/api/chat', methods=['POST'])
def api_chat():
    """API endpoint implementing an improved RAG logic using schema + sampled table data."""
    global qa_chain
    user_message = request.json.get('message', '').strip()

    if qa_chain is None:
        return jsonify({"success": False, "message": "AI bot not initialized. Check API key and database files."}), 503
    if not user_message:
        return jsonify({"error": "Empty message"}), 400

    # Ensure schema/docs are loaded
    db_schema = get_db_schema()

    # Retrieve best-matching table docs to provide targeted context
    relevant_docs = retrieve_relevant_docs(user_message, top_k=3)
    context_snippets = "\n\n".join(f"{d['table']} ({d['db']}): {d['sample_text'] or d['schema']}" for d in relevant_docs)

    # --- Step 1: AI generates a safe SQL Query (SELECT only) ---
    sql_prompt = f"""
You are an expert SQL analyst for an agricultural SQLite database. Based on the user's question and the provided database context, produce a single, runnable SELECT-only SQLite query (no explanations). Keep results concise (LIMIT 20). If multiple tables are relevant, prefer queries that use the most appropriate table(s) listed.

DATABASE SCHEMAS (short):
{db_schema}

RELEVANT TABLE EXTRACTS:
{context_snippets}

IMPORTANT RULES:
- OUTPUT ONLY the single, runnable SQL SELECT query and nothing else.
- Do NOT include any commentary, backticks, or markup.
- Enforce LIMIT 20.
- Use double quotes for identifiers that contain special characters or spaces.
- Do not run any data modification commands (INSERT/UPDATE/DELETE/PRAGMA/etc).

User question: {user_message}

SQL Query:
"""
    try:
        sql_response = qa_chain.generate_content(sql_prompt)
        sql_query = sql_response.text.strip()
        # strip any trailing semicolons and extra text; keep only first SELECT statement
        sql_query = re.split(r';|\n\n', sql_query)[0].strip()
        # ensure it's a SELECT
        if not re.match(r'^\s*SELECT\b', sql_query, re.IGNORECASE):
            return jsonify({"success": False, "message": "The model did not return a safe SELECT query. Please rephrase."}), 400

        # ensure LIMIT exists; if not, append LIMIT 20
        if not re.search(r'\bLIMIT\b', sql_query, re.IGNORECASE):
            sql_query = sql_query.rstrip(';') + " LIMIT 20"

        # --- Step 2: Execute SQL Query and Retrieve Data Context ---
        # prefer database inferred from top relevant doc if any
        prefer_db_path = None
        if relevant_docs:
            prefer_db_path = relevant_docs[0].get('path')

        data_context = get_data_for_rag(sql_query, prefer_db=prefer_db_path)

        # --- Step 3: AI generates Final Answer based on retrieved data ---
        final_answer_prompt = f"""
You are an expert agricultural consultant for AgriBase. Use the RETRIEVED DATA below to answer the user's question succinctly and helpfully. Use precise figures where possible and be explicit if numbers are forecasts/predictions. Mention data quality briefly when data looks noisy or is sampled from PDF extractions.

User question: {user_message}

SQL used: {sql_query}

RETRIEVED DATA (use this as factual basis; if it's an error or empty, say so):
{data_context}

INSTRUCTIONS:
- Provide a short, actionable answer (3-6 sentences).
- Cite specific numbers or trends that come from the RETRIEVED DATA.
- If data is from a predictions/forecast table, clearly label it a forecast.
- If data is noisy or ambiguous, mention uncertainty and recommend "verify with raw data or re-run query".
- Offer one practical suggestion or next step for the user.

Answer:
"""
        final_response = qa_chain.generate_content(final_answer_prompt)
        answer = final_response.text.strip()

        return jsonify({"success": True, "message": answer, "sql_used": sql_query, "context_snippets": context_snippets})

    except Exception as e:
        print(f"[ERROR] Chatbot processing failed: {str(e)}")
        return jsonify({
            "success": False,
            "message": "I apologize, an internal processing error occurred. This might be due to an un-runnable SQL query or API issue. Please try rephrasing your question."
        }), 500

# ...existing code (other routes) ...
if __name__ == '__main__':
    initialize_qa_chain()
    app.run(debug=True)
