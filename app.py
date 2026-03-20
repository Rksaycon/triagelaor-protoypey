"""
TriageLaor ML Backend
Features:
  1. Panic Input Normalization
  2. Typing Speed Analysis
  3. Credibility / Prank Detection
  4. Category + Severity Classification
  5. Severity Mismatch Detection
  6. Priority Score
  7. Escalation Detection
  8. Time-Decay Urgency

Run: python app.py
  Citizen:  http://127.0.0.1:5000/
  Official: http://127.0.0.1:5000/dashboard
"""

import pickle, json, os, re, time, uuid, urllib.request, urllib.parse, sqlite3
from datetime import datetime
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# ── Gemini API Key ─────────────────────────────────────────────────────────────
GEMINI_API_KEY = 'AIzaSyDE7FDIkHb-9HT4fAIUu-YzrxWuMnIx_JQ'
GEMINI_URL = f'https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}'

# ── SQLite Database ────────────────────────────────────────────────────────────
DB_PATH = os.path.join(os.path.dirname(__file__), 'triage.db')

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    conn.execute('''
        CREATE TABLE IF NOT EXISTS reports (
            id              TEXT PRIMARY KEY,
            description     TEXT,
            normalized      TEXT,
            location_text   TEXT,
            lat             REAL,
            lng             REAL,
            category        TEXT,
            severity        TEXT,
            credibility     TEXT,
            credibility_score REAL,
            priority_score  INTEGER,
            escalated       INTEGER DEFAULT 0,
            typing_profile  TEXT,
            eta_label       TEXT,
            status          TEXT DEFAULT "Pending",
            timestamp       REAL,
            time_label      TEXT,
            dispatched_to   TEXT DEFAULT NULL,
            dispatched_at   TEXT DEFAULT NULL
        )
    ''')
    # ── Migration: add columns if they don't exist (for existing DBs) ─────────
    existing = [row[1] for row in conn.execute('PRAGMA table_info(reports)').fetchall()]
    if 'dispatched_to' not in existing:
        conn.execute('ALTER TABLE reports ADD COLUMN dispatched_to TEXT DEFAULT NULL')
        print("✅ Migrated: added dispatched_to column")
    if 'dispatched_at' not in existing:
        conn.execute('ALTER TABLE reports ADD COLUMN dispatched_at TEXT DEFAULT NULL')
        print("✅ Migrated: added dispatched_at column")
    conn.execute('''
        CREATE TABLE IF NOT EXISTS responders (
            id      INTEGER PRIMARY KEY AUTOINCREMENT,
            name    TEXT NOT NULL,
            role    TEXT NOT NULL,
            phone   TEXT NOT NULL,
            active  INTEGER DEFAULT 1
        )
    ''')
    # Seed default responders if empty
    count = conn.execute('SELECT COUNT(*) FROM responders').fetchone()[0]
    if count == 0:
        default_responders = [
            ('Juan dela Cruz',    'Tanod Commander',      '09171234567'),
            ('Maria Santos',      'Health Worker',        '09181234567'),
            ('Pedro Reyes',       'BDRRMC Coordinator',   '09191234567'),
            ('Ana Garcia',        'Barangay Tanod',       '09201234567'),
            ('Jose Mendoza',      'Rescue Team Lead',     '09211234567'),
        ]
        conn.executemany(
            'INSERT INTO responders (name, role, phone) VALUES (?,?,?)',
            default_responders
        )
        print("✅ Default responders seeded")
    conn.commit()
    conn.close()
    print("✅ Database ready:", DB_PATH)

init_db()

# ── DB helpers ─────────────────────────────────────────────────────────────────
def load_reports_from_db():
    """Load all reports from DB into the in-memory REPORTS dict on startup."""
    conn = get_db()
    rows = conn.execute('SELECT * FROM reports').fetchall()
    conn.close()
    result = {}
    for row in rows:
        r = dict(row)
        r['escalated'] = bool(r['escalated'])
        r.setdefault('dispatched_to', None)
        r.setdefault('dispatched_at', None)
        result[r['id']] = r
    return result

def save_report_to_db(report: dict):
    conn = get_db()
    conn.execute('''
        INSERT OR REPLACE INTO reports
        (id, description, normalized, location_text, lat, lng,
         category, severity, credibility, credibility_score,
         priority_score, escalated, typing_profile, eta_label,
         status, timestamp, time_label)
        VALUES
        (:id, :description, :normalized, :location_text, :lat, :lng,
         :category, :severity, :credibility, :credibility_score,
         :priority_score, :escalated, :typing_profile, :eta_label,
         :status, :timestamp, :time_label)
    ''', {**report, 'escalated': int(report.get('escalated', False))})
    conn.commit()
    conn.close()

def update_report_status_db(report_id: str, status: str, eta_label: str = None):
    conn = get_db()
    if eta_label:
        conn.execute(
            'UPDATE reports SET status=?, eta_label=? WHERE id=?',
            (status, eta_label, report_id)
        )
    else:
        conn.execute(
            'UPDATE reports SET status=? WHERE id=?',
            (status, report_id)
        )
    conn.commit()
    conn.close()

def delete_report_from_db(report_id: str):
    conn = get_db()
    conn.execute('DELETE FROM reports WHERE id=?', (report_id,))
    conn.commit()
    conn.close()

# ── Semaphore SMS ──────────────────────────────────────────────────────────────
# Sign up free at semaphore.co — replace with your real API key to send real SMS
# Leave as 'SIMULATE' to just log the SMS without sending
SEMAPHORE_API_KEY = 'SIMULATE'
SEMAPHORE_SENDER  = 'TRIAGE'

def send_sms(phone: str, message: str) -> dict:
    """
    Sends SMS via Semaphore Philippine SMS gateway.
    If SEMAPHORE_API_KEY is 'SIMULATE', just logs the message.
    """
    if SEMAPHORE_API_KEY == 'SIMULATE':
        print(f"\n📱 SMS SIMULATED → {phone}")
        print(f"   {message}\n")
        return {'status': 'simulated', 'phone': phone, 'message': message}

    try:
        payload = urllib.parse.urlencode({
            'apikey':      SEMAPHORE_API_KEY,
            'number':      phone,
            'message':     message,
            'sendername':  SEMAPHORE_SENDER,
        }).encode('utf-8')

        req = urllib.request.Request(
            'https://api.semaphore.co/api/v4/messages',
            data=payload,
            method='POST'
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            result = json.loads(resp.read().decode())
            print(f"✅ SMS sent to {phone}")
            return {'status': 'sent', 'result': result}
    except Exception as e:
        print(f"❌ SMS failed: {e}")
        return {'status': 'failed', 'error': str(e)}


def build_dispatch_sms(report: dict, responder: dict) -> str:
    """Builds the SMS message sent to the responder."""
    loc = report.get('location_text') or ''
    gps = ''
    if report.get('lat') and report.get('lng'):
        gps = f"\nGPS: {report['lat']:.5f},{report['lng']:.5f}"
        gps += f"\nMap: https://maps.google.com/?q={report['lat']},{report['lng']}"

    return (
        f"[TRIAGE-LAOR] DISPATCH ALERT\n"
        f"RPT-{report['id']} | {report['category'].upper()} | {report['severity'].upper()}\n"
        f"Location: {loc or 'See GPS below'}{gps}\n"
        f"Report: {report['description'][:100]}\n"
        f"Respond immediately. -BDRRMC Del Rosario"
    )
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')
try:
    with open(os.path.join(MODEL_DIR, 'category_model.pkl'), 'rb') as f:
        category_model = pickle.load(f)
    with open(os.path.join(MODEL_DIR, 'severity_model.pkl'), 'rb') as f:
        severity_model = pickle.load(f)
    with open(os.path.join(MODEL_DIR, 'credibility_model.pkl'), 'rb') as f:
        credibility_model = pickle.load(f)
    MODEL_LOADED = True
    print("✅ All models loaded")
except FileNotFoundError:
    MODEL_LOADED = False
    print("⚠️  Run: python train_model.py first")

# ── In-memory report store (loaded from DB on startup) ────────────────────────
REPORTS = load_reports_from_db()
print(f"✅ Loaded {len(REPORTS)} reports from database")

# ══════════════════════════════════════════════════════════════════════════════
# PANIC NORMALIZER
# ══════════════════════════════════════════════════════════════════════════════
def normalize_panic(text: str) -> str:
    text = re.sub(r'([a-zA-Z])\1+', r'\1', text)
    text = re.sub(r'[!?]{2,}', '!', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text.lower()

# ══════════════════════════════════════════════════════════════════════════════
# TYPING SPEED ANALYZER
# ══════════════════════════════════════════════════════════════════════════════
def analyze_typing(typing_data: dict) -> dict:
    if not typing_data:
        return {'panic_score': 0.5, 'cps': 0, 'profile': 'unknown',
                'backspace_ratio': 0, 'pauses': 0}
    total_ms   = max(typing_data.get('total_time_ms', 1), 1)
    chars      = max(typing_data.get('char_count', 1), 1)
    backspaces = typing_data.get('backspace_count', 0)
    pauses     = typing_data.get('pause_count', 0)
    cps = chars / (total_ms / 1000)
    backspace_ratio = backspaces / chars

    if cps > 5 and backspace_ratio < 0.15:
        profile, panic_score = 'panic', 0.9
    elif cps > 3.5 and backspace_ratio < 0.25:
        profile, panic_score = 'urgent', 0.75
    elif cps < 1.2 and pauses > 2:
        profile, panic_score = 'deliberate', 0.1
    elif cps < 2 and pauses > 1:
        profile, panic_score = 'casual', 0.2
    else:
        profile, panic_score = 'normal', 0.5

    return {
        'cps': round(cps, 2),
        'backspace_ratio': round(backspace_ratio, 2),
        'pauses': pauses,
        'profile': profile,
        'panic_score': panic_score,
    }

# ══════════════════════════════════════════════════════════════════════════════
# PRIORITY SCORE
# ══════════════════════════════════════════════════════════════════════════════
CATEGORY_WEIGHTS = {
    'Medical Emergency': 40,
    'Fire': 35,
    'Flooding': 30,
    'Infrastructure Damage': 20,
}
SEVERITY_WEIGHTS = {'High': 40, 'Medium': 25, 'Low': 10}

# ══════════════════════════════════════════════════════════════════════════════
# ETA ESTIMATOR
# ══════════════════════════════════════════════════════════════════════════════

ETA_BASE = {
    ('Medical Emergency', 'High'):   (5,  10,  'minutes'),
    ('Medical Emergency', 'Medium'): (15, 30,  'minutes'),
    ('Medical Emergency', 'Low'):    (30, 60,  'minutes'),
    ('Fire',              'High'):   (10, 15,  'minutes'),
    ('Fire',              'Medium'): (20, 40,  'minutes'),
    ('Fire',              'Low'):    (30, 60,  'minutes'),
    ('Flooding',          'High'):   (15, 30,  'minutes'),
    ('Flooding',          'Medium'): (30, 60,  'minutes'),
    ('Flooding',          'Low'):    (1,  3,   'hours'),
    ('Infrastructure Damage', 'High'):   (30, 60, 'minutes'),
    ('Infrastructure Damage', 'Medium'): (2,  6,  'hours'),
    ('Infrastructure Damage', 'Low'):    (1,  3,  'days'),
}

def compute_eta(category: str, severity: str) -> dict:
    key = (category, severity)
    low, high, unit = ETA_BASE.get(key, (1, 3, 'hours'))

    # Queue factor — count pending High/Medium reports ahead
    pending_high = sum(
        1 for r in REPORTS.values()
        if r['status'] == 'Pending'
        and r['severity'] in ('High', 'Medium')
        and r['credibility'] == 'credible'
    )

    # Add queue delay only for minutes-based ETAs
    queue_delay = 0
    if unit == 'minutes' and pending_high > 3:
        queue_delay = (pending_high - 3) * 5  # +5 min per extra report
        low  += queue_delay
        high += queue_delay

    # Build label
    if unit == 'minutes':
        label = f'~{low}–{high} minutes'
    elif unit == 'hours':
        label = f'~{low}–{high} hours'
    else:
        label = f'~{low}–{high} days'

    # Urgency message
    if severity == 'High':
        msg = 'BDRRMC officials have been alerted. Response is being dispatched.'
    elif severity == 'Medium':
        msg = 'Your report is queued for response. Officials are aware.'
    else:
        msg = 'Your report is logged and will be addressed in routine operations.'

    return {
        'eta_label': label,
        'unit': unit,
        'low': low,
        'high': high,
        'queue_delay_added': queue_delay,
        'pending_ahead': pending_high,
        'message': msg,
    }


def compute_priority(category, severity, credibility_score, escalated=False, decay_boost=0):
    base = CATEGORY_WEIGHTS.get(category, 20) + SEVERITY_WEIGHTS.get(severity, 10)
    cred_bonus = int(credibility_score * 20)
    escalation_bonus = 15 if escalated else 0
    total = base + cred_bonus + escalation_bonus + decay_boost
    return min(total, 100)

# ══════════════════════════════════════════════════════════════════════════════
# SEVERITY MISMATCH
# ══════════════════════════════════════════════════════════════════════════════
def check_severity_mismatch(user_selected, ml_predicted):
    order = {'High': 3, 'Medium': 2, 'Low': 1}
    u, m = order.get(user_selected, 0), order.get(ml_predicted, 0)
    if not u or not m:
        return {'mismatch': False}
    diff = m - u
    if diff >= 1:
        return {'mismatch': True, 'direction': 'underreported',
                'message': f'Your description suggests {ml_predicted.upper()} severity, not {user_selected.upper()}. Please verify.'}
    elif diff <= -1:
        return {'mismatch': True, 'direction': 'overreported',
                'message': f'Your description suggests {ml_predicted.upper()} severity. Are you sure it is {user_selected.upper()}?'}
    return {'mismatch': False}

# ══════════════════════════════════════════════════════════════════════════════
# ESCALATION DETECTION
# Checks if 3+ reports of same category came in within 30 minutes
# ══════════════════════════════════════════════════════════════════════════════
ESCALATION_THRESHOLD = 3      # number of reports
ESCALATION_WINDOW    = 30 * 60  # 30 minutes in seconds

def check_escalation(category: str) -> dict:
    now = time.time()
    window_start = now - ESCALATION_WINDOW

    # Count recent reports with same category
    recent = [
        r for r in REPORTS.values()
        if r['category'] == category
        and r['timestamp'] >= window_start
        and r['credibility'] == 'credible'
    ]

    count = len(recent)
    if count >= ESCALATION_THRESHOLD:
        return {
            'escalated': True,
            'count': count,
            'message': f'⚠ ESCALATED — {count} residents reported {category} in the last 30 minutes',
        }
    return {
        'escalated': False,
        'count': count,
        'message': f'{count} similar report(s) in last 30 min',
    }

# ══════════════════════════════════════════════════════════════════════════════
# TIME-DECAY URGENCY
# Unresolved High reports get +5 per 30 min, max +20
# ══════════════════════════════════════════════════════════════════════════════
def compute_decay_boost(severity: str, timestamp: float) -> int:
    if severity != 'High':
        return 0
    elapsed_min = (time.time() - timestamp) / 60
    intervals = int(elapsed_min / 30)
    return min(intervals * 5, 20)

# ══════════════════════════════════════════════════════════════════════════════
# ROUTES
# ══════════════════════════════════════════════════════════════════════════════

@app.route('/')
def index():
    return render_template('citizen.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'models_loaded': MODEL_LOADED,
                    'total_reports': len(REPORTS)})

@app.route('/predict', methods=['POST'])
def predict():
    if not MODEL_LOADED:
        return jsonify({'error': 'Models not loaded. Run train_model.py first.'}), 503

    data = request.get_json(silent=True)
    if not data or 'description' not in data:
        return jsonify({'error': 'Missing description'}), 400

    raw          = str(data.get('description', '')).strip()
    user_severity = str(data.get('user_severity', '')).strip()
    typing_data  = data.get('typing_data', {})
    lat          = data.get('lat', None)
    lng          = data.get('lng', None)
    location_text = str(data.get('location_text', '')).strip()

    if not raw:
        return jsonify({'error': 'Empty description'}), 400

    # 1. Normalize panic text
    normalized    = normalize_panic(raw)
    was_normalized = normalized != raw.lower()
    text_for_ml   = normalized or raw.lower()

    # 2. Typing analysis
    typing = analyze_typing(typing_data)

    # 3. Credibility
    cred_pred  = credibility_model.predict([text_for_ml])[0]
    cred_proba = credibility_model.predict_proba([text_for_ml])[0]
    cred_scores = dict(zip(credibility_model.classes_, cred_proba.tolist()))
    text_cred   = float(cred_scores.get('credible', 0.5))
    blended_cred = round(min(max(text_cred * 0.7 + typing['panic_score'] * 0.3, 0.0), 1.0), 3)
    final_cred   = 'credible' if blended_cred >= 0.5 else 'suspicious'
    is_suspicious = final_cred == 'suspicious'

    # 4. Classify
    cat_pred   = category_model.predict([text_for_ml])[0]
    cat_proba  = category_model.predict_proba([text_for_ml])[0]
    cat_scores = dict(zip(category_model.classes_, cat_proba.tolist()))

    sev_pred   = severity_model.predict([text_for_ml])[0]
    sev_proba  = severity_model.predict_proba([text_for_ml])[0]
    sev_scores = dict(zip(severity_model.classes_, sev_proba.tolist()))

    # 5. Escalation
    escalation = check_escalation(cat_pred)

    # 6. Priority
    priority = compute_priority(cat_pred, sev_pred, blended_cred,
                                 escalated=escalation['escalated'])

    # 7. Mismatch
    mismatch = check_severity_mismatch(user_severity, sev_pred) if user_severity else {}

    # NOTE: No ETA on submission — citizen polls for status update from official

    # 8. Store report
    ts = time.time()
    report_id = str(uuid.uuid4())[:8].upper()
    REPORTS[report_id] = {
        'id':               report_id,
        'description':      raw,
        'normalized':       normalized,
        'location_text':    location_text,
        'lat':              lat,
        'lng':              lng,
        'category':         cat_pred,
        'severity':         sev_pred,
        'credibility':      final_cred,
        'credibility_score': blended_cred,
        'priority_score':   priority,
        'escalated':        escalation['escalated'],
        'typing_profile':   typing['profile'],
        'status':           'Pending',
        'eta_label':        None,
        'timestamp':        ts,
        'time_label':       datetime.fromtimestamp(ts).strftime('%H:%M:%S'),
    }
    save_report_to_db(REPORTS[report_id])

    return jsonify({
        'report_id':        report_id,
        'original_input':   raw,
        'normalized_input': normalized,
        'was_normalized':   was_normalized,
        'category':         cat_pred,
        'severity':         sev_pred,
        'credibility':      final_cred,
        'credibility_score': blended_cred,
        'is_suspicious':    is_suspicious,
        'priority_score':   priority,
        'escalation':       escalation,
        'mismatch':         mismatch,
        'typing':           typing,
        'category_scores':  {k: round(v, 3) for k, v in cat_scores.items()},
        'severity_scores':  {k: round(v, 3) for k, v in sev_scores.items()},
    })


@app.route('/track/<report_id>')
def track(report_id):
    return render_template('track.html')


@app.route('/reports/<report_id>/citizen', methods=['GET'])
def citizen_status(report_id):
    """
    Citizen polls this endpoint to check if official has responded.
    Returns status + ETA once official marks as Ongoing.
    """
    if report_id not in REPORTS:
        return jsonify({'error': 'Report not found'}), 404

    r = REPORTS[report_id]
    status = r['status']

    response = {
        'report_id':      report_id,
        'status':         status,
        'category':       r['category'],
        'severity':       r['severity'],
        'credibility':    r['credibility'],
        'priority_score': r['priority_score'],
        'description':    r['description'],
        'location_text':  r.get('location_text', ''),
        'lat':            r.get('lat'),
        'lng':            r.get('lng'),
        'time_label':     r.get('time_label', ''),
        'eta':            None,
        'message':        None,
    }

    if status == 'Pending':
        response['message'] = 'Your report is awaiting review by a barangay official.'

    elif status == 'Ongoing':
        eta = compute_eta(r['category'], r['severity'])
        response['eta'] = eta
        response['message'] = eta['message']

    elif status == 'Resolved':
        response['message'] = 'Your report has been resolved by barangay officials. Thank you.'

    return jsonify(response)


@app.route('/reports', methods=['GET'])
def get_reports():
    """Returns all reports sorted by priority (with time-decay applied)."""
    result = []
    for r in REPORTS.values():
        decay = compute_decay_boost(r['severity'], r['timestamp'])
        current_priority = min(r['priority_score'] + decay, 100)
        elapsed_min = int((time.time() - r['timestamp']) / 60)
        result.append({**r,
                        'current_priority': current_priority,
                        'decay_boost': decay,
                        'elapsed_min': elapsed_min})

    result.sort(key=lambda x: x['current_priority'], reverse=True)
    return jsonify(result)


@app.route('/reports/<report_id>/status', methods=['POST'])
def update_status(report_id):
    """Official updates report status. ETA is computed when set to Ongoing."""
    if report_id not in REPORTS:
        return jsonify({'error': 'Report not found'}), 404
    data = request.get_json(silent=True) or {}
    new_status = data.get('status', 'Pending')
    REPORTS[report_id]['status'] = new_status

    # Compute and store ETA the moment official marks as Ongoing
    if new_status == 'Ongoing':
        r = REPORTS[report_id]
        eta = compute_eta(r['category'], r['severity'])
        REPORTS[report_id]['eta_label'] = eta['eta_label']
        update_report_status_db(report_id, new_status, eta['eta_label'])
    elif new_status == 'Resolved':
        REPORTS[report_id]['eta_label'] = 'Resolved'
        update_report_status_db(report_id, new_status, 'Resolved')
    else:
        update_report_status_db(report_id, new_status)

    return jsonify({'ok': True, 'status': new_status})


@app.route('/reports/<report_id>', methods=['DELETE'])
def delete_report(report_id):
    if report_id in REPORTS:
        del REPORTS[report_id]
    delete_report_from_db(report_id)
    return jsonify({'ok': True})


@app.route('/summary', methods=['GET'])
def get_summary():
    """
    Uses Gemini to generate a full AI tactical briefing for the official dashboard.
    """
    now = time.time()

    if not REPORTS:
        return jsonify({
            'summary': 'No active reports on record. Monitoring systems operational. All clear.',
            'status': 'clear',
            'stats': {'pending': 0, 'ongoing': 0, 'high': 0, 'escalated': 0, 'suspicious': 0}
        })

    # Build detailed report list for Gemini
    report_lines = []
    for r in sorted(REPORTS.values(), key=lambda x: x['priority_score'], reverse=True):
        elapsed_min = int((now - r['timestamp']) / 60)
        loc = r.get('location_text') or ('GPS pinned' if r.get('lat') else 'location unknown')
        line = (
            f"RPT-{r['id']} | {r['status'].upper()} | {r['category']} | "
            f"{r['severity']} severity | Priority {r['priority_score']}/100 | "
            f"Pending {elapsed_min} min | "
            f"{'⚠ ESCALATED ({} similar reports) | '.format(r.get('escalation_count','')) if r['escalated'] else ''}"
            f"{'🚨 SUSPICIOUS/PRANK FLAGGED | ' if r['credibility'] == 'suspicious' else ''}"
            f"Location: {loc} | "
            f"GPS: {'YES ('+str(round(r['lat'],4))+','+str(round(r['lng'],4))+')' if r.get('lat') else 'NO'} | "
            f"Typing profile: {r.get('typing_profile','unknown')} | "
            f"Description: {r['description'][:120]}"
        )
        report_lines.append(line)

    reports_text = '\n'.join(report_lines)

    # Stats
    total    = len(REPORTS)
    pending  = sum(1 for r in REPORTS.values() if r['status'] == 'Pending')
    ongoing  = sum(1 for r in REPORTS.values() if r['status'] == 'Ongoing')
    resolved = sum(1 for r in REPORTS.values() if r['status'] == 'Resolved')
    high     = sum(1 for r in REPORTS.values() if r['severity'] == 'High' and r['status'] != 'Resolved')
    medium   = sum(1 for r in REPORTS.values() if r['severity'] == 'Medium' and r['status'] != 'Resolved')
    esc      = sum(1 for r in REPORTS.values() if r['escalated'])
    susp     = sum(1 for r in REPORTS.values() if r['credibility'] == 'suspicious')
    flood    = sum(1 for r in REPORTS.values() if r['category'] == 'Flooding' and r['status'] != 'Resolved')
    fire     = sum(1 for r in REPORTS.values() if r['category'] == 'Fire' and r['status'] != 'Resolved')
    medical  = sum(1 for r in REPORTS.values() if r['category'] == 'Medical Emergency' and r['status'] != 'Resolved')
    infra    = sum(1 for r in REPORTS.values() if r['category'] == 'Infrastructure Damage' and r['status'] != 'Resolved')

    current_time = datetime.now().strftime('%I:%M %p')

    prompt = f"""You are TRIAGE-AI, the emergency command intelligence system for Barangay Del Rosario, Milaor, Camarines Sur, Philippines. You support the BDRRMC (Barangay Disaster Risk Reduction and Management Council) during active emergencies.

TIME: {current_time}
BARANGAY: Del Rosario, Milaor, Camarines Sur

ACTIVE INCIDENT REPORTS ({total} total):
{reports_text}

SUMMARY STATS:
- Status: {pending} pending, {ongoing} ongoing, {resolved} resolved
- Severity: {high} HIGH, {medium} MEDIUM active
- Escalated clusters: {esc}
- Suspicious/prank flagged: {susp}
- By category: Flooding={flood}, Fire={fire}, Medical={medical}, Infrastructure={infra}

You must generate a FULL TACTICAL BRIEFING for the barangay official. This is a real emergency command system — people's lives may depend on your analysis.

Structure your response EXACTLY like this (no markdown, plain text, use ALL CAPS for section headers):

SITUATION OVERVIEW
Write 2-3 sentences. What is the overall situation right now? Is it escalating or stable? How serious is it on a scale?

HIGHEST PRIORITY INCIDENTS
List the top 2-3 most critical incidents the official must act on RIGHT NOW. For each one say: what it is, where it is, how long it has been waiting, and why it is urgent.

PATTERN ANALYSIS
Are multiple reports coming from the same area suggesting a widespread event? Are there suspicious reports mixed in? What does the typing behavior data tell us (panic vs deliberate)?

RECOMMENDED ACTIONS
Give 3-5 specific, numbered, actionable orders the official should execute right now. Be specific — name the category, the location, the resource to deploy.

WATCH LIST
Any reports that need monitoring but are not yet critical. Any suspicious reports the official should be aware of.

Keep the total response under 280 words. Be direct. Be urgent when the situation is urgent. This is not a chatbot — this is a command system."""

    try:
        body = json.dumps({
            'contents': [{'parts': [{'text': prompt}]}],
            'generationConfig': {
                'maxOutputTokens': 500,
                'temperature': 0.3
            }
        }).encode('utf-8')

        req = urllib.request.Request(
            GEMINI_URL,
            data=body,
            headers={'Content-Type': 'application/json'},
            method='POST'
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode())
            text = data['candidates'][0]['content']['parts'][0]['text'].strip()

        # Determine alert level
        if high >= 3 or esc >= 2 or flood >= 3:
            status = 'critical'
        elif high >= 1 or esc >= 1 or pending >= 5:
            status = 'warning'
        else:
            status = 'normal'

        return jsonify({
            'summary': text,
            'status':  status,
            'stats': {
                'pending':    pending,
                'ongoing':    ongoing,
                'resolved':   resolved,
                'high':       high,
                'escalated':  esc,
                'suspicious': susp,
                'flood':      flood,
                'fire':       fire,
                'medical':    medical,
                'infra':      infra,
            }
        })

    except Exception as e:
        # Fallback if Gemini fails
        if high >= 3 or esc >= 2:
            fb_status = 'critical'
            fb_msg = (
                f"CRITICAL SITUATION — {high} HIGH severity incidents active, {esc} escalated. "
                f"IMMEDIATE RESPONSE REQUIRED. "
                f"{flood} flooding, {fire} fire, {medical} medical, {infra} infrastructure reports pending. "
                f"Dispatch resources immediately. {susp} suspicious reports flagged — verify before acting."
            )
        elif high >= 1:
            fb_status = 'warning'
            fb_msg = (
                f"ACTIVE EMERGENCY — {pending} pending, {high} HIGH severity unresolved. "
                f"Categories: Flooding={flood}, Fire={fire}, Medical={medical}, Infra={infra}. "
                f"Review priority-ranked reports. {susp} suspicious flagged."
            )
        else:
            fb_status = 'normal'
            fb_msg = (
                f"SITUATION STABLE — {total} total reports, {pending} pending. "
                f"No critical incidents. Monitor ongoing reports."
            )
        return jsonify({
            'summary': fb_msg,
            'status':  fb_status,
            'error':   str(e),
            'stats': {
                'pending': pending, 'ongoing': ongoing, 'high': high,
                'escalated': esc, 'suspicious': susp
            }
        })


@app.route('/responder/sms')
def responder_sms():
    """Simulated SMS view for the responder — looks like a real phone message."""
    return render_template('responder.html')


@app.route('/responders', methods=['GET'])
def get_responders():
    """Returns list of all active responders."""
    conn = get_db()
    rows = conn.execute(
        'SELECT * FROM responders WHERE active=1 ORDER BY name'
    ).fetchall()
    conn.close()
    return jsonify([dict(r) for r in rows])


@app.route('/responders', methods=['POST'])
def add_responder():
    """Add a new responder."""
    data = request.get_json(silent=True) or {}
    name  = str(data.get('name', '')).strip()
    role  = str(data.get('role', '')).strip()
    phone = str(data.get('phone', '')).strip()
    if not name or not role or not phone:
        return jsonify({'error': 'name, role, and phone are required'}), 400
    conn = get_db()
    conn.execute(
        'INSERT INTO responders (name, role, phone) VALUES (?,?,?)',
        (name, role, phone)
    )
    conn.commit()
    conn.close()
    return jsonify({'ok': True, 'message': f'{name} added as responder'})


@app.route('/responders/<int:responder_id>', methods=['DELETE'])
def remove_responder(responder_id):
    """Deactivate a responder."""
    conn = get_db()
    conn.execute('UPDATE responders SET active=0 WHERE id=?', (responder_id,))
    conn.commit()
    conn.close()
    return jsonify({'ok': True})


@app.route('/reports/<report_id>/dispatch', methods=['POST'])
def dispatch_report(report_id):
    """
    Official dispatches a report to a responder.
    Sends SMS to the responder's phone automatically.
    """
    if report_id not in REPORTS:
        return jsonify({'error': 'Report not found'}), 404

    data = request.get_json(silent=True) or {}
    responder_id = data.get('responder_id')
    if not responder_id:
        return jsonify({'error': 'responder_id required'}), 400

    # Get responder
    conn = get_db()
    responder = conn.execute(
        'SELECT * FROM responders WHERE id=? AND active=1',
        (responder_id,)
    ).fetchone()
    conn.close()

    if not responder:
        return jsonify({'error': 'Responder not found'}), 404

    responder = dict(responder)
    report    = REPORTS[report_id]

    # Build and send SMS
    sms_text   = build_dispatch_sms(report, responder)
    sms_result = send_sms(responder['phone'], sms_text)

    # Update report
    dispatched_at = datetime.now().strftime('%H:%M:%S')
    REPORTS[report_id]['dispatched_to'] = responder['name']
    REPORTS[report_id]['dispatched_at'] = dispatched_at
    REPORTS[report_id]['status']        = 'Ongoing'

    # Save to DB
    conn = get_db()
    conn.execute(
        '''UPDATE reports
           SET dispatched_to=?, dispatched_at=?, status="Ongoing"
           WHERE id=?''',
        (responder['name'], dispatched_at, report_id)
    )
    conn.commit()
    conn.close()

    return jsonify({
        'ok':            True,
        'dispatched_to': responder['name'],
        'phone':         responder['phone'],
        'role':          responder['role'],
        'sms_status':    sms_result['status'],
        'sms_preview':   sms_text,
        'sms_view_url':  f"/responder/sms?report={report_id}&name={responder['name']}",
    })


if __name__ == '__main__':
    print("\n🚀 TriageLaor ML Server")
    print("   Citizen:  http://127.0.0.1:5000/")
    print("   Official: http://127.0.0.1:5000/dashboard\n")
    app.run(host='0.0.0.0', port=5000, debug=False)