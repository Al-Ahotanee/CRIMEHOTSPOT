╔══════════════════════════════════════════════════════════════════════════════╗
║  SENTINEL-NW  |  Enterprise Crime Hotspot Intelligence Platform             ║
║  NW Nigeria — Katsina · Zamfara · Sokoto · Kaduna                          ║
║  Flask + SQLite + GradientBoosting + ACLED                                 ║
╚══════════════════════════════════════════════════════════════════════════════╝

from flask import Flask, jsonify, request, send_from_directory, g
from flask_cors import CORS
import sqlite3, os, json, hashlib, secrets, time, threading
from datetime import datetime, timedelta
from functools import wraps
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.ensemble import GradientBoostingClassifier
import warnings
warnings.filterwarnings("ignore")

# ─── CONFIG ───────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
# PRODUCTION READY: Allows mounting a Render Persistent Disk via Environment Variables to save DB across restarts
DB_PATH     = os.environ.get("DB_PATH", os.path.join(BASE_DIR, "sentinel.db"))
# PRODUCTION READY: Use fixed secret key from env to prevent session loss on Render restarts
SECRET_KEY  = os.environ.get("SECRET_KEY", secrets.token_hex(32))
TOKEN_TTL   = 3600  # 1 hour session
MAX_REQUESTS_PER_MIN = 60

app = Flask(__name__, static_folder=".", static_url_path="")
CORS(app, resources={r"/api/*": {"origins": "*"}})
app.config["SECRET_KEY"] = SECRET_KEY

# ─── GLOBAL STATE ─────────────────────────────────────────────────────────────
_models   = {}
_scaler   = None
_le_state = None
_df       = None
_feature_cols = []
_reports  = {}
_lock     = threading.Lock()
_rate_cache = {}  # ip -> [timestamps]

# ═══════════════════════════════════════════════════════════════════════════════
# DATABASE INIT
# ═══════════════════════════════════════════════════════════════════════════════
def get_db():
    db = getattr(g, "_database", None)
    if db is None:
        db = g._database = sqlite3.connect(DB_PATH, check_same_thread=False)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA journal_mode=WAL")
        db.execute("PRAGMA foreign_keys=ON")
    return db

@app.teardown_appcontext
def close_db(exception):
    db = getattr(g, "_database", None)
    if db is not None:
        db.close()

def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            username    TEXT    UNIQUE NOT NULL,
            password_hash TEXT  NOT NULL,
            role        TEXT    DEFAULT 'analyst',
            clearance   TEXT    DEFAULT 'SECRET',
            full_name   TEXT,
            unit        TEXT,
            created_at  TEXT    DEFAULT (datetime('now')),
            last_login  TEXT,
            is_active   INTEGER DEFAULT 1
        );

        CREATE TABLE IF NOT EXISTS sessions (
            token       TEXT    PRIMARY KEY,
            user_id     INTEGER NOT NULL,
            created_at  TEXT    DEFAULT (datetime('now')),
            expires_at  TEXT    NOT NULL,
            ip_address  TEXT,
            user_agent  TEXT,
            FOREIGN KEY (user_id) REFERENCES users(id)
        );

        CREATE TABLE IF NOT EXISTS audit_log (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id     INTEGER,
            action      TEXT    NOT NULL,
            resource    TEXT,
            ip_address  TEXT,
            payload     TEXT,
            timestamp   TEXT    DEFAULT (datetime('now')),
            success     INTEGER DEFAULT 1
        );

        CREATE TABLE IF NOT EXISTS incidents (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            event_date      TEXT,
            event_type      TEXT,
            sub_event_type  TEXT,
            actor1          TEXT,
            admin1          TEXT,
            admin2          TEXT,
            location        TEXT,
            latitude        REAL,
            longitude       REAL,
            fatalities      INTEGER DEFAULT 0,
            is_kidnapping   INTEGER DEFAULT 0,
            is_bandit_terror INTEGER DEFAULT 0,
            event_severity  INTEGER DEFAULT 1,
            grid_id         TEXT,
            source          TEXT    DEFAULT 'ACLED',
            notes           TEXT,
            created_at      TEXT    DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS risk_assessments (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id     INTEGER,
            state       TEXT,
            latitude    REAL,
            longitude   REAL,
            month       INTEGER,
            year        INTEGER,
            kidnap_prob REAL,
            bandit_prob REAL,
            overall_risk REAL,
            risk_label  TEXT,
            created_at  TEXT DEFAULT (datetime('now')),
            FOREIGN KEY (user_id) REFERENCES users(id)
        );

        CREATE TABLE IF NOT EXISTS alerts (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            title       TEXT    NOT NULL,
            severity    TEXT    NOT NULL,
            state       TEXT,
            event_type  TEXT,
            description TEXT,
            is_read     INTEGER DEFAULT 0,
            created_at  TEXT    DEFAULT (datetime('now'))
        );

        CREATE INDEX IF NOT EXISTS idx_incidents_state     ON incidents(admin1);
        CREATE INDEX IF NOT EXISTS idx_incidents_date      ON incidents(event_date);
        CREATE INDEX IF NOT EXISTS idx_incidents_grid      ON incidents(grid_id);
        CREATE INDEX IF NOT EXISTS idx_sessions_token      ON sessions(token);
        CREATE INDEX IF NOT EXISTS idx_audit_user          ON audit_log(user_id);
        """)

        # Seed default admin user (password: Sentinel@2025)
        pw_hash = hashlib.sha256("Sentinel@2025".encode()).hexdigest()
        try:
            conn.execute("""
                INSERT OR IGNORE INTO users (username,password_hash,role,clearance,full_name,unit)
                VALUES (?,?,?,?,?,?)
            """, ("admin", pw_hash, "admin", "TOP SECRET", "System Administrator", "SENTINEL HQ"))
            conn.execute("""
                INSERT OR IGNORE INTO users (username,password_hash,role,clearance,full_name,unit)
                VALUES (?,?,?,?,?,?)
            """, ("analyst", hashlib.sha256("Analyst@2025".encode()).hexdigest(),
                  "analyst", "SECRET", "Intel Analyst", "NW Nigeria Desk"))
        except:
            pass
        conn.commit()

# ═══════════════════════════════════════════════════════════════════════════════
# ML PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════
def generate_synthetic_data():
    np.random.seed(42)
    n = 10000
    state_coords = {
        "Katsina": (12.98, 7.60), "Zamfara": (12.17, 6.23),
        "Sokoto":  (13.06, 5.24), "Kaduna":  (10.52, 7.44),
        "Kebbi":   (12.45, 4.20), "Jigawa":  (12.23, 9.35),
    }
    event_types = ["Battles","Violence against civilians",
                   "Explosions/Remote violence","Riots",
                   "Strategic developments","Protests"]
    sub_event_map = {
        "Battles":                    ["Armed clash","Government regains territory","Non-state actor overtakes territory"],
        "Violence against civilians": ["Attack","Abduction/forced disappearance","Sexual violence"],
        "Explosions/Remote violence": ["Suicide bomb","Remote explosive/landmine/IED","Air/drone strike","Shelling/artillery/missile attack"],
        "Riots":                      ["Mob violence","Violent demonstration"],
        "Strategic developments":     ["Looting/property destruction","Agreement","Arrests"],
        "Protests":                   ["Peaceful protest","Excessive force against protesters"],
    }
    actors = ["Bandits","Boko Haram","ISWAP","Fulani Militia",
              "Government Forces (Nigeria)","Unknown Armed Group","Civilian Joint Task Force"]
    states = np.random.choice(list(state_coords.keys()), n, p=[0.22,0.25,0.18,0.20,0.08,0.07])
    lats, lons = [], []
    for s in states:
        clat, clon = state_coords[s]
        lats.append(clat + np.random.normal(0, 0.6))
        lons.append(clon + np.random.normal(0, 0.6))
    chosen_events = np.random.choice(event_types, n, p=[0.30,0.35,0.12,0.08,0.10,0.05])
    sub_events = [np.random.choice(sub_event_map[e]) for e in chosen_events]
    months = np.random.choice(range(1,13), n,
             p=[0.12,0.11,0.10,0.06,0.05,0.04,0.05,0.06,0.07,0.08,0.11,0.15])
    years  = np.random.choice(range(2019,2025), n)
    days   = np.random.randint(1, 28, n)
    fatalities = np.where(
        np.isin(chosen_events, ["Battles","Violence against civilians","Explosions/Remote violence"]),
        np.random.negative_binomial(2, 0.4, n), np.zeros(n, dtype=int))
    return pd.DataFrame({
        "event_date":     [f"{y}-{m:02d}-{d:02d}" for y,m,d in zip(years,months,days)],
        "event_type":     chosen_events,
        "sub_event_type": sub_events,
        "actor1":         np.random.choice(actors, n),
        "admin1":         states,
        "latitude":       lats,
        "longitude":      lons,
        "fatalities":     fatalities,
    })

def init_ml():
    global _models, _scaler, _le_state, _df, _feature_cols, _reports
    print("🧠 Initialising ML pipeline...")
    df = generate_synthetic_data()
    df["event_date"] = pd.to_datetime(df["event_date"], errors="coerce")
    df.dropna(subset=["event_date","latitude","longitude"], inplace=True)
    df["year"]  = df["event_date"].dt.year
    df["month"] = df["event_date"].dt.month
    df["day"]   = df["event_date"].dt.day
    df["dow"]   = df["event_date"].dt.dayofweek
    df["quarter"]       = df["event_date"].dt.quarter
    df["is_dry_season"] = df["month"].isin([11,12,1,2,3]).astype(int)
    df["is_weekend"]    = (df["dow"] >= 5).astype(int)
    GRID = 0.2
    df["lat_bin"] = (df["latitude"]  / GRID).round(0) * GRID
    df["lon_bin"] = (df["longitude"] / GRID).round(0) * GRID
    df["grid_id"] = df["lat_bin"].astype(str) + "_" + df["lon_bin"].astype(str)
    event_enc = {"Battles":5,"Violence against civilians":4,
                 "Explosions/Remote violence":5,"Riots":2,
                 "Strategic developments":1,"Protests":1}
    df["event_severity"]  = df["event_type"].map(event_enc).fillna(1)
    df["is_kidnapping"]   = df["sub_event_type"].str.contains(
        "Abduction|kidnap|forced disappearance", case=False, na=False).astype(int)
    df["is_bandit_terror"]= (
        df["actor1"].str.contains("Bandit|ISWAP|Boko|militia|armed group", case=False, na=False) |
        df["event_type"].isin(["Battles","Explosions/Remote violence"])
    ).astype(int)
    df.sort_values("event_date", inplace=True)
    df["fatalities"] = pd.to_numeric(df["fatalities"], errors="coerce").fillna(0)
    grid_counts = df.groupby("grid_id").size().reset_index(name="grid_total_incidents")
    df = df.merge(grid_counts, on="grid_id", how="left")
    state_month = df.groupby(["admin1","year","month"]).size().reset_index(name="state_month_count")
    df = df.merge(state_month, on=["admin1","year","month"], how="left")
    grid_fat = df.groupby("grid_id")["fatalities"].mean().reset_index(name="avg_fatalities_grid")
    df = df.merge(grid_fat, on="grid_id", how="left")
    le = LabelEncoder()
    df["state_enc"] = le.fit_transform(df["admin1"].fillna("Unknown"))
    feature_cols = ["latitude","longitude","lat_bin","lon_bin","year","month","day",
                    "dow","quarter","is_dry_season","is_weekend","event_severity",
                    "fatalities","grid_total_incidents","state_month_count",
                    "avg_fatalities_grid","state_enc"]
    TARGET_COLS = ["is_kidnapping","is_bandit_terror"]
    scaler = StandardScaler()
    df_ml  = df[feature_cols + TARGET_COLS].dropna()
    X = scaler.fit_transform(df_ml[feature_cols])
    models, reports = {}, {}
    for target in TARGET_COLS:
        y = df_ml[target]
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        model = GradientBoostingClassifier(n_estimators=300, max_depth=6, learning_rate=0.05,
                                           subsample=0.8, random_state=42)
        model.fit(X_tr, y_tr)
        preds = model.predict(X_te)
        reports[target] = classification_report(y_te, preds, output_dict=True)
        models[target]  = model
    with _lock:
        _models, _scaler, _le_state = models, scaler, le
        _df, _feature_cols, _reports = df, feature_cols, reports

    # Persist incidents to SQLite
    persist_incidents(df)
    generate_alerts(df)
    print(f"✅ ML ready. {len(df)} events loaded.")

def persist_incidents(df):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("DELETE FROM incidents")
        rows = df[["event_date","event_type","sub_event_type","actor1","admin1",
                    "latitude","longitude","fatalities","is_kidnapping",
                    "is_bandit_terror","event_severity","grid_id"]].copy()
        rows["event_date"] = rows["event_date"].astype(str)
        rows.to_sql("incidents", conn, if_exists="append", index=False)

def generate_alerts(df):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("DELETE FROM alerts")
        recent = df[df["year"] >= 2023]
        hot_states = recent.groupby("admin1").size().nlargest(3)
        for state, cnt in hot_states.items():
            sev = "CRITICAL" if cnt > 600 else "HIGH"
            conn.execute("""
                INSERT INTO alerts (title,severity,state,event_type,description)
                VALUES (?,?,?,?,?)
            """, (f"Elevated Activity — {state}", sev, state, "Multi-type",
                  f"{cnt} incidents recorded in {state} since 2023. Heightened vigilance advised."))
        # Bandit surge alert
        bandit_surge = recent[recent["is_bandit_terror"]==1].groupby("admin1").size().nlargest(2)
        for state, cnt in bandit_surge.items():
            conn.execute("""
                INSERT INTO alerts (title,severity,state,event_type,description)
                VALUES (?,?,?,?,?)
            """, (f"Bandit Surge Detected — {state}", "HIGH", state, "Bandit/Terror",
                  f"Bandit/terrorist incidents spiked to {cnt} in {state}. Intelligence assets activated."))
        conn.commit()

# ═══════════════════════════════════════════════════════════════════════════════
# AUTH MIDDLEWARE
# ═══════════════════════════════════════════════════════════════════════════════
def hash_password(pw):
    return hashlib.sha256(pw.encode()).hexdigest()

def create_token(user_id, ip, ua):
    token = secrets.token_urlsafe(48)
    exp   = (datetime.utcnow() + timedelta(seconds=TOKEN_TTL)).isoformat()
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            INSERT INTO sessions (token,user_id,expires_at,ip_address,user_agent)
            VALUES (?,?,?,?,?)
        """, (token, user_id, exp, ip, ua))
    return token

def verify_token(token):
    if not token:
        return None
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("""
            SELECT s.user_id, s.expires_at, u.username, u.role, u.clearance, u.full_name, u.unit
            FROM sessions s JOIN users u ON s.user_id=u.id
            WHERE s.token=? AND u.is_active=1
        """, (token,)).fetchone()
    if not row:
        return None
    if datetime.fromisoformat(row["expires_at"]) < datetime.utcnow():
        return None
    return dict(row)

def require_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get("X-Auth-Token") or request.args.get("token")
        user  = verify_token(token)
        if not user:
            return jsonify({"error":"Unauthorized","code":401}), 401
        request.current_user = user
        return f(*args, **kwargs)
    return decorated

def require_admin(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get("X-Auth-Token")
        user  = verify_token(token)
        if not user or user["role"] != "admin":
            return jsonify({"error":"Forbidden","code":403}), 403
        request.current_user = user
        return f(*args, **kwargs)
    return decorated

def rate_limit(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        ip  = request.remote_addr
        now = time.time()
        hits = [t for t in _rate_cache.get(ip,[]) if now-t < 60]
        hits.append(now)
        _rate_cache[ip] = hits
        if len(hits) > MAX_REQUESTS_PER_MIN:
            return jsonify({"error":"Rate limit exceeded","code":429}), 429
        return f(*args, **kwargs)
    return decorated

def audit(action, resource=None, payload=None, success=True):
    try:
        user = getattr(request, "current_user", None)
        uid  = user["user_id"] if user else None
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("""
                INSERT INTO audit_log (user_id,action,resource,ip_address,payload,success)
                VALUES (?,?,?,?,?,?)
            """, (uid, action, resource, request.remote_addr,
                  json.dumps(payload) if payload else None, int(success)))
    except:
        pass

# ═══════════════════════════════════════════════════════════════════════════════
# ROUTES — AUTH
# ═══════════════════════════════════════════════════════════════════════════════
@app.route("/api/auth/login", methods=["POST"])
@rate_limit
def login():
    data = request.json or {}
    username = (data.get("username","")).strip()
    password = data.get("password","")
    if not username or not password:
        return jsonify({"error":"Credentials required"}), 400
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        user = conn.execute(
            "SELECT * FROM users WHERE username=? AND is_active=1", (username,)
        ).fetchone()
    if not user or user["password_hash"] != hash_password(password):
        audit("LOGIN_FAIL", resource=username, success=False)
        return jsonify({"error":"Invalid credentials"}), 401
    token = create_token(user["id"], request.remote_addr,
                         request.headers.get("User-Agent",""))
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("UPDATE users SET last_login=datetime('now') WHERE id=?", (user["id"],))
    audit("LOGIN", resource=username)
    return jsonify({
        "token": token,
        "user": {
            "id": user["id"], "username": user["username"],
            "role": user["role"], "clearance": user["clearance"],
            "full_name": user["full_name"], "unit": user["unit"]
        },
        "expires_in": TOKEN_TTL
    })

@app.route("/api/auth/logout", methods=["POST"])
@require_auth
def logout():
    token = request.headers.get("X-Auth-Token")
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("DELETE FROM sessions WHERE token=?", (token,))
    audit("LOGOUT")
    return jsonify({"message":"Logged out"})

@app.route("/api/auth/me", methods=["GET"])
@require_auth
def me():
    return jsonify(request.current_user)

# ═══════════════════════════════════════════════════════════════════════════════
# ROUTES — DASHBOARD / STATS
# ═══════════════════════════════════════════════════════════════════════════════
@app.route("/api/dashboard/summary", methods=["GET"])
@require_auth
@rate_limit
def dashboard_summary():
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        total   = conn.execute("SELECT COUNT(*) as c FROM incidents").fetchone()["c"]
        kidnaps = conn.execute("SELECT COUNT(*) as c FROM incidents WHERE is_kidnapping=1").fetchone()["c"]
        bandits = conn.execute("SELECT COUNT(*) as c FROM incidents WHERE is_bandit_terror=1").fetchone()["c"]
        fatal   = conn.execute("SELECT SUM(fatalities) as s FROM incidents").fetchone()["s"] or 0
        by_state= conn.execute("""
            SELECT admin1, COUNT(*) as total,
                   SUM(is_kidnapping) as kidnappings,
                   SUM(is_bandit_terror) as bandit_terror,
                   SUM(fatalities) as fatalities
            FROM incidents GROUP BY admin1 ORDER BY total DESC
        """).fetchall()
        by_month= conn.execute("""
            SELECT substr(event_date,1,7) as month, COUNT(*) as total,
                   SUM(is_kidnapping) as kidnappings,
                   SUM(is_bandit_terror) as bandit_terror
            FROM incidents WHERE event_date IS NOT NULL
            GROUP BY month ORDER BY month
        """).fetchall()
        by_type = conn.execute("""
            SELECT event_type, COUNT(*) as total
            FROM incidents GROUP BY event_type ORDER BY total DESC
        """).fetchall()
        by_actor= conn.execute("""
            SELECT actor1, COUNT(*) as total, SUM(fatalities) as fatalities
            FROM incidents GROUP BY actor1 ORDER BY total DESC LIMIT 8
        """).fetchall()
        alerts_count = conn.execute(
            "SELECT COUNT(*) as c FROM alerts WHERE is_read=0").fetchone()["c"]
        recent_assess = conn.execute("""
            SELECT COUNT(*) as c FROM risk_assessments
            WHERE created_at >= datetime('now','-7 days')
        """).fetchone()["c"]

    audit("VIEW_DASHBOARD")
    return jsonify({
        "totals": {"incidents": total, "kidnappings": kidnaps,
                   "bandit_terror": bandits, "fatalities": int(fatal),
                   "unread_alerts": alerts_count, "assessments_7d": recent_assess},
        "by_state": [dict(r) for r in by_state],
        "by_month": [dict(r) for r in by_month],
        "by_type":  [dict(r) for r in by_type],
        "by_actor": [dict(r) for r in by_actor],
    })

# ═══════════════════════════════════════════════════════════════════════════════
# ROUTES — INCIDENTS
# ═══════════════════════════════════════════════════════════════════════════════
@app.route("/api/incidents", methods=["GET"])
@require_auth
@rate_limit
def get_incidents():
    page     = max(1, int(request.args.get("page", 1)))
    per_page = min(200, int(request.args.get("per_page", 50)))
    state    = request.args.get("state")
    etype    = request.args.get("event_type")
    kid_only = request.args.get("kidnapping") == "1"
    ban_only = request.args.get("bandit")     == "1"
    offset   = (page - 1) * per_page

    where, params = ["1=1"], []
    if state:    where.append("admin1=?");           params.append(state)
    if etype:    where.append("event_type=?");       params.append(etype)
    if kid_only: where.append("is_kidnapping=1")
    if ban_only: where.append("is_bandit_terror=1")

    sql = f"""
        SELECT * FROM incidents WHERE {' AND '.join(where)}
        ORDER BY event_date DESC LIMIT ? OFFSET ?
    """
    cnt_sql = f"SELECT COUNT(*) as c FROM incidents WHERE {' AND '.join(where)}"

    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows  = conn.execute(sql, params + [per_page, offset]).fetchall()
        total = conn.execute(cnt_sql, params).fetchone()["c"]

    return jsonify({
        "incidents": [dict(r) for r in rows],
        "total": total, "page": page, "per_page": per_page,
        "pages": (total + per_page - 1) // per_page
    })

@app.route("/api/incidents/heatmap", methods=["GET"])
@require_auth
@rate_limit
def heatmap_data():
    state  = request.args.get("state")
    etype  = request.args.get("event_type","All")
    where, params = ["latitude IS NOT NULL AND longitude IS NOT NULL"], []
    if state and state != "All": where.append("admin1=?"); params.append(state)
    if etype == "Kidnappings":        where.append("is_kidnapping=1")
    elif etype == "Bandit/Terror":    where.append("is_bandit_terror=1")

    sql = f"""
        SELECT latitude, longitude, fatalities, event_type, admin1,
               event_date, actor1
        FROM incidents WHERE {' AND '.join(where)} LIMIT 5000
    """
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(sql, params).fetchall()

    return jsonify({"points": [dict(r) for r in rows]})

@app.route("/api/incidents/hotgrids", methods=["GET"])
@require_auth
def hot_grids():
    state = request.args.get("state")
    where, params = ["1=1"], []
    if state and state != "All": where.append("admin1=?"); params.append(state)
    sql = f"""
        SELECT grid_id, admin1,
               AVG(latitude) as lat, AVG(longitude) as lon,
               COUNT(*) as total, SUM(fatalities) as fatalities,
               SUM(is_kidnapping) as kidnappings
        FROM incidents WHERE {' AND '.join(where)}
        GROUP BY grid_id ORDER BY total DESC LIMIT 20
    """
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(sql, params).fetchall()
    return jsonify({"grids": [dict(r) for r in rows]})

# ═══════════════════════════════════════════════════════════════════════════════
# ROUTES — RISK PREDICTION
# ═══════════════════════════════════════════════════════════════════════════════
@app.route("/api/predict", methods=["POST"])
@require_auth
@rate_limit
def predict():
    data   = request.json or {}
    state  = data.get("state","Katsina")
    month  = int(data.get("month", datetime.now().month))
    year   = int(data.get("year",  datetime.now().year))
    lat    = float(data.get("latitude",  12.98))
    lon    = float(data.get("longitude",  7.60))
    fatal  = int(data.get("fatalities", 2))

    with _lock:
        if not _models:
            return jsonify({"error":"ML models not ready"}), 503
        GRID = 0.2
        lat_b   = round(lat / GRID) * GRID
        lon_b   = round(lon / GRID) * GRID
        grid_id = f"{lat_b}_{lon_b}"
        g_total = int((_df["grid_id"] == grid_id).sum())
        s_m     = int(((_df["admin1"]==state) & (_df["month"]==month)).sum())
        avg_fat = float(_df[_df["grid_id"]==grid_id]["fatalities"].mean() or 0)
        state_enc_val = int(_le_state.transform([state])[0]) if state in _le_state.classes_ else 0

        row_vals = [lat, lon, lat_b, lon_b, year, month, 15, 2,
                    (month-1)//3+1, int(month in [11,12,1,2,3]), 0, 3,
                    fatal, g_total, s_m, avg_fat, state_enc_val]
        X_in = _scaler.transform([row_vals])

        results = {}
        for target, mdl in _models.items():
            results[target] = round(float(mdl.predict_proba(X_in)[0][1]) * 100, 1)

    k_risk = results["is_kidnapping"]
    b_risk = results["is_bandit_terror"]
    overall = round((k_risk * 0.5 + b_risk * 0.5), 1)

    def label(p):
        if p >= 70: return "CRITICAL"
        if p >= 50: return "HIGH"
        if p >= 30: return "MODERATE"
        return "LOW"

    # Save assessment
    user = request.current_user
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            INSERT INTO risk_assessments
            (user_id,state,latitude,longitude,month,year,kidnap_prob,bandit_prob,overall_risk,risk_label)
            VALUES (?,?,?,?,?,?,?,?,?,?)
        """, (user["user_id"], state, lat, lon, month, year,
              k_risk, b_risk, overall, label(overall)))

    audit("PREDICT_RISK", payload={"state":state,"month":month,"year":year,"overall":overall})
    return jsonify({
        "state": state, "month": month, "year": year,
        "latitude": lat, "longitude": lon,
        "kidnap_probability":  k_risk,
        "bandit_probability":  b_risk,
        "overall_risk":        overall,
        "risk_label":          label(overall),
        "advisory": (
            "CRITICAL — Avoid all non-essential travel. Request armed escort." if overall >= 70 else
            "HIGH — High risk zone. Security coordination mandatory." if overall >= 50 else
            "MODERATE — Exercise caution. Monitor intelligence feeds." if overall >= 30 else
            "LOW — Standard security protocols apply."
        ),
        "breakdown": {
            "kidnapping":  {"probability": k_risk,  "label": label(k_risk)},
            "bandit_terror":{"probability": b_risk, "label": label(b_risk)},
        }
    })

# ═══════════════════════════════════════════════════════════════════════════════
# ROUTES — ALERTS
# ═══════════════════════════════════════════════════════════════════════════════
@app.route("/api/alerts", methods=["GET"])
@require_auth
def get_alerts():
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM alerts ORDER BY created_at DESC LIMIT 50"
        ).fetchall()
    return jsonify({"alerts": [dict(r) for r in rows]})

@app.route("/api/alerts/<int:alert_id>/read", methods=["PATCH"])
@require_auth
def mark_read(alert_id):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("UPDATE alerts SET is_read=1 WHERE id=?", (alert_id,))
    return jsonify({"success": True})

# ═══════════════════════════════════════════════════════════════════════════════
# ROUTES — ADMIN
# ═══════════════════════════════════════════════════════════════════════════════
@app.route("/api/admin/users", methods=["GET"])
@require_admin
def list_users():
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT id,username,role,clearance,full_name,unit,created_at,last_login,is_active FROM users"
        ).fetchall()
    return jsonify({"users": [dict(r) for r in rows]})

@app.route("/api/admin/users", methods=["POST"])
@require_admin
def create_user():
    data = request.json or {}
    required = ["username","password","role","clearance"]
    if not all(data.get(k) for k in required):
        return jsonify({"error":"Missing required fields"}), 400
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("""
                INSERT INTO users (username,password_hash,role,clearance,full_name,unit)
                VALUES (?,?,?,?,?,?)
            """, (data["username"], hash_password(data["password"]),
                  data["role"], data["clearance"],
                  data.get("full_name",""), data.get("unit","")))
        audit("CREATE_USER", payload={"username":data["username"]})
        return jsonify({"success": True}), 201
    except sqlite3.IntegrityError:
        return jsonify({"error":"Username already exists"}), 409

@app.route("/api/admin/users/<int:uid>/toggle", methods=["PATCH"])
@require_admin
def toggle_user(uid):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("UPDATE users SET is_active=1-is_active WHERE id=?", (uid,))
    audit("TOGGLE_USER", payload={"uid": uid})
    return jsonify({"success": True})

@app.route("/api/admin/audit", methods=["GET"])
@require_admin
def audit_log():
    page = int(request.args.get("page",1))
    per  = 50
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(f"""
            SELECT a.*, u.username FROM audit_log a
            LEFT JOIN users u ON a.user_id=u.id
            ORDER BY a.timestamp DESC LIMIT {per} OFFSET {(page-1)*per}
        """).fetchall()
        total = conn.execute("SELECT COUNT(*) as c FROM audit_log").fetchone()["c"]
    return jsonify({"logs": [dict(r) for r in rows], "total": total})

@app.route("/api/model/performance", methods=["GET"])
@require_auth
def model_perf():
    perf = {}
    for tgt, rpt in _reports.items():
        perf[tgt] = {
            "accuracy":  round(rpt.get("accuracy",0)*100, 1),
            "precision": round(rpt.get("1",{}).get("precision",0)*100, 1),
            "recall":    round(rpt.get("1",{}).get("recall",   0)*100, 1),
            "f1":        round(rpt.get("1",{}).get("f1-score", 0)*100, 1),
        }
    return jsonify({"performance": perf, "feature_cols": _feature_cols})

@app.route("/api/risk/history", methods=["GET"])
@require_auth
def risk_history():
    user = request.current_user
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("""
            SELECT * FROM risk_assessments WHERE user_id=?
            ORDER BY created_at DESC LIMIT 20
        """, (user["user_id"],)).fetchall()
    return jsonify({"history": [dict(r) for r in rows]})

# ═══════════════════════════════════════════════════════════════════════════════
# STATIC — serve SPA
# ═══════════════════════════════════════════════════════════════════════════════
@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve_spa(path):
    if path.startswith("api/"):
        return jsonify({"error":"Not found"}), 404
    return send_from_directory(".", "index.html")

# ═══════════════════════════════════════════════════════════════════════════════
# PRODUCTION BOOTSTRAP (GUNICORN / RENDER WSGI SUPPORT)
# ═══════════════════════════════════════════════════════════════════════════════
def startup():
    print("╔══════════════════════════════════════════╗")
    print("║   SENTINEL-NW  |  Intelligence Platform  ║")
    print("╚══════════════════════════════════════════╝")
    init_db()
    # Train ML in background so server starts fast in production WSGI environments
    ml_thread = threading.Thread(target=init_ml, daemon=True)
    ml_thread.start()

# This is executed instantly when the module is imported by Gunicorn
startup()

# ═══════════════════════════════════════════════════════════════════════════════
# LOCAL BOOT (For local development via python app.py)
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    # Ensure Render's dynamically assigned PORT is used
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)