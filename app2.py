"""
Flask app reading trial.db (default) and exposing APIs plus simple crop-analysis UI.

Features:
- Uses trial.db created with update_trial_db.py
- /api/crops : list crops with has_variety flag and table_count
- /api/tables/<crop> : tables for crop
- /api/table/<table_id> : rows (list of JSON objects)
- /api/varieties/<crop> : union of variety values across tables for crop (used to show variety selector only when present)
- UI /crop_analysis shows table selector and shows variety selector only if varieties exist
"""
from flask import Flask, jsonify, request, render_template_string
import sqlite3
import json
from pathlib import Path

DB_PATH = "trial.db"  # uses trial.db to avoid touching agri-base.db

app = Flask(__name__)


def get_db_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


@app.route("/api/crops", methods=["GET"])
def api_crops():
    conn = get_db_conn()
    cur = conn.cursor()
    cur.execute("""
      SELECT lower(crop) AS crop,
             MAX(COALESCE(has_variety,0)) AS has_variety,
             COUNT(*) AS table_count
      FROM tables_meta
      WHERE crop IS NOT NULL
      GROUP BY lower(crop)
      ORDER BY lower(crop)
    """)
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return jsonify(rows)


@app.route("/api/tables/<crop>", methods=["GET"])
def api_tables_for_crop(crop):
    conn = get_db_conn()
    cur = conn.cursor()
    cur.execute("""
      SELECT id, filename, page, descriptor, scope, has_variety, created_at
      FROM tables_meta
      WHERE lower(crop) = lower(?)
      ORDER BY COALESCE(page, 9999), id
    """, (crop,))
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return jsonify(rows)


@app.route("/api/table/<int:table_id>", methods=["GET"])
def api_table_rows(table_id):
    conn = get_db_conn()
    cur = conn.cursor()
    cur.execute("SELECT row_json FROM crop_data WHERE table_id = ? ORDER BY row_index", (table_id,))
    rows = []
    for r in cur.fetchall():
        try:
            rows.append(json.loads(r[0]))
        except Exception:
            rows.append(r[0])
    conn.close()
    return jsonify(rows)


@app.route("/api/varieties/<crop>", methods=["GET"])
def api_varieties_for_crop(crop):
    """
    Return sorted unique variety strings found across all tables for a crop.
    Looks for keys containing 'variet' or 'var' or 'hybrid' in row JSON keys/values.
    """
    conn = get_db_conn()
    cur = conn.cursor()
    cur.execute("SELECT id FROM tables_meta WHERE lower(crop) = lower(?) AND has_variety = 1", (crop,))
    table_ids = [r[0] for r in cur.fetchall()]
    variants = set()
    for tid in table_ids:
        cur.execute("SELECT row_json FROM crop_data WHERE table_id = ?", (tid,))
        for r in cur.fetchall():
            try:
                od = json.loads(r[0])
            except Exception:
                continue
            for k, v in od.items():
                if v is None:
                    continue
                kl = k.lower()
                vl = str(v).strip()
                if any(tok in kl for tok in ('variet', 'varian', 'var.','hybrid')):
                    if vl:
                        variants.add(vl)
                else:
                    # sometimes variety appears as value in other columns
                    if re_search_variety(vl):
                        variants.add(vl)
    conn.close()
    out = sorted(variants)
    return jsonify(out)


def re_search_variety(s: str) -> bool:
    s = (s or "").lower()
    return bool(s and any(tok in s for tok in ('variet', 'varian', 'hybrid', 'local', 'hyb')))


# Minimal UI
UI_HTML = """
<!doctype html>
<html>
<head><title>Crop Analysis</title></head>
<body>
  <h2>Crop Analysis (trial.db)</h2>
  <div>
    <label>Crop:</label>
    <select id="crop_select" onchange="onCrop()"></select>
  </div>
  <div>
    <label>Table:</label>
    <select id="table_select" onchange="onTable()"></select>
  </div>
  <div id="variety_div" style="display:none;">
    <label>Variety:</label>
    <select id="var_select"></select>
  </div>
  <div>
    <button onclick="loadTable()">Load Table</button>
  </div>
  <pre id="result"></pre>

<script>
async function loadCrops(){
  const res = await fetch('/api/crops');
  const data = await res.json();
  const sel = document.getElementById('crop_select');
  sel.innerHTML = '<option value="">--select--</option>';
  data.forEach(c => {
    const opt = document.createElement('option');
    opt.value = c.crop;
    opt.text = c.crop + ' (' + c.table_count + ')';
    opt.dataset.hasVar = c.has_variety;
    sel.appendChild(opt);
  });
}
async function onCrop(){
  const crop = document.getElementById('crop_select').value;
  const sel = document.getElementById('table_select');
  sel.innerHTML = '';
  document.getElementById('result').textContent = '';
  document.getElementById('var_select').innerHTML = '';
  document.getElementById('variety_div').style.display = 'none';
  if(!crop) return;
  const res = await fetch('/api/tables/' + encodeURIComponent(crop));
  const tables = await res.json();
  tables.forEach(t => {
    const opt = document.createElement('option');
    opt.value = t.id;
    opt.text = '['+ (t.page || '?') + '] ' + t.filename + (t.descriptor ? ' :: ' + t.descriptor : '');
    opt.dataset.hasVar = t.has_variety;
    sel.appendChild(opt);
  });
  // if crop-level variety info exists, fetch varieties and show selector
  const vres = await fetch('/api/varieties/' + encodeURIComponent(crop));
  const variants = await vres.json();
  const vd = document.getElementById('variety_div');
  const varsel = document.getElementById('var_select');
  varsel.innerHTML = '';
  if(variants && variants.length){
    vd.style.display = 'block';
    variants.forEach(v => {
      const o = document.createElement('option');
      o.value = v; o.text = v;
      varsel.appendChild(o);
    });
  } else {
    vd.style.display = 'none';
  }
}
function onTable(){
  // optional: do something when a specific table selected
}
async function loadTable(){
  const sel = document.getElementById('table_select');
  const id = sel.value;
  if(!id) return;
  const res = await fetch('/api/table/' + id);
  const rows = await res.json();
  document.getElementById('result').textContent = JSON.stringify(rows, null, 2);
}
window.onload = loadCrops;
</script>
</body>
</html>
"""

@app.route("/crop_analysis")
def crop_analysis_ui():
    return render_template_string(UI_HTML)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
