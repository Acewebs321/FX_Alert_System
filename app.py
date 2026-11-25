import requests
import pandas as pd
import numpy as np
import time
import os
import threading
import sqlite3
import joblib
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from collections import deque
from flask import Flask, jsonify, request, render_template_string



# CONFIGURATION & GLOBAL STATE
# ==============================
# Both pairs are scanned independently with their own ML model when available.
app_state = {
    "symbol": "EUR_JPY",  # Dashboard display selection only; both pairs are always scanned.
    "oanda_health": "Waiting...", "telegram_health": "Waiting...",
    "scanner_status": "Starting up...", "m5_trend": "Unknown",
    "ml_status": "Checking model...", "todays_news": [], "pairs": {}
}



# CONFIGURATION & GLOBAL STATE
# ==============================
# Locked to EUR_JPY for the EUR_JPY-trained ML Revisit Model
app_state = {
    "symbol": "EUR_JPY",
    "oanda_health": "Waiting...",
    "telegram_health": "Waiting...",
    "scanner_status": "Starting up...",
    "m5_trend": "Unknown",
    "ml_status": "Checking model...",
    "todays_news": []
}



OANDA_URL = "https://api-fxpractice.oanda.com"
TELEGRAM_TOKEN = "8641583463:AAGrsLR0Xmf-fOd3FXafjC8lbqUiqCasZGM"
TELEGRAM_CHAT_ID ="1110417286"
OANDA_TOKEN = "56c3289f164a1a64b30de44241b6ec5a-4aa1d8f08ad01339e4ee2f4d6be3d7c9"
# Supports both EUR_JPY and GBP_JPY with dedicated pair-specific ML models/thresholds
AVAILABLE_PAIRS = ["EUR_JPY", "GBP_JPY"]
SCAN_INTERVAL = 20
FVG_LOOKBACK = 10
M3_SMA_PERIOD = 20
M5_SMA_PERIOD = 50
FVG_LOOKAHEAD = 10
M5_CACHE_SECONDS = 60
MAX_SETUP_AGE_SECONDS = 30 * 60  # Never retain/act on a setup older than 30 minutes.
DB_CLEANUP_INTERVAL_SECONDS = 5 * 60
ALERT_DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "alert_deduplication.sqlite3")
MODEL_PATHS = {
    "EUR_JPY": ["fvg_revisit_model_EUR_JPY.joblib", "fvg_revisit_model.joblib"],
    "GBP_JPY": ["fvg_revisit_model_GBP_JPY.joblib"]
}
PAIR_DEFAULTS = {
    "EUR_JPY": {"q25_gap": 0.006, "q50_gap": 0.012, "q75_gap": 0.024},
    "GBP_JPY": {"q25_gap": 0.008, "q50_gap": 0.015, "q75_gap": 0.030}
}
news_cache = {"timestamp": None, "data": []}
pending_crossings = {pair: {} for pair in AVAILABLE_PAIRS}
last_processed_crossing = {pair: None for pair in AVAILABLE_PAIRS}
sending_setups = set()
alert_lock = threading.Lock()
m5_cache = {pair: {"timestamp": 0.0, "trend": None, "sma50": None} for pair in AVAILABLE_PAIRS}
ml_bundles = {pair: None for pair in AVAILABLE_PAIRS}
pair_status = {
    pair: {"oanda_health": "Waiting...", "scanner_status": "Starting up...",
           "m5_trend": "Unknown", "ml_status": "Checking model...",
           "last_candle": None, "last_alert": None}
    for pair in AVAILABLE_PAIRS
}
MAX_LOGS = 100
system_logs = deque(maxlen=MAX_LOGS)
app = Flask(__name__)

# ==============================
# LOGGING SYSTEM
# ==============================
def log_msg(msg: str):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted_msg = f"[{timestamp}] {msg}"
    print(formatted_msg, flush=True)
    system_logs.append(formatted_msg)

# ==============================
# ML MODEL LOADER & PREDICTOR
# ==============================
def load_ml_models():
    loaded_pairs = []
    for pair, candidate_files in MODEL_PATHS.items():
        ml_bundles[pair] = None
        for path in candidate_files:
            if os.path.exists(path):
                try:
                    ml_bundles[pair] = joblib.load(path)
                    loaded_pairs.append(f"{pair} ({path})")
                    log_msg(f"Loaded ML model for {pair} from {path}.")
                    break
                except Exception as e:
                    log_msg(f"Error loading {path} for {pair}: {e}")
    update_ml_status_badge()

def update_ml_status_badge():
    for pair in AVAILABLE_PAIRS:
        pair_status[pair]["ml_status"] = (
            f"Loaded ({pair}) 🟢" if ml_bundles.get(pair) is not None
            else f"Stats Fallback ({pair}) 🟡"
        )
    app_state["ml_status"] = "Both pair models checked"


def initialise_alert_db():
    """Create/migrate the SQLite de-duplication table with a 30-minute TTL."""
    with sqlite3.connect(ALERT_DB_PATH, timeout=10) as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS alert_setups (
                setup_id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                updated_at REAL NOT NULL,
                created_at REAL
            )
        """)
        columns = {
            row[1] for row in connection.execute("PRAGMA table_info(alert_setups)")
        }
        if "created_at" not in columns:
            connection.execute("ALTER TABLE alert_setups ADD COLUMN created_at REAL")
            # Existing records pre-date the TTL column; updated_at is the safest
            # available timestamp for assigning their initial expiry.
            connection.execute(
                "UPDATE alert_setups SET created_at = updated_at WHERE created_at IS NULL"
            )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_alert_setups_created_at "
            "ON alert_setups(created_at)"
        )
        connection.commit()
    cleanup_expired_setup_records()


def cleanup_expired_setup_records(now=None):
    """Delete SQLite setup records once they have been stored for 30 minutes."""
    now = time.time() if now is None else now
    cutoff = now - MAX_SETUP_AGE_SECONDS
    with sqlite3.connect(ALERT_DB_PATH, timeout=10) as connection:
        cursor = connection.execute(
            "DELETE FROM alert_setups WHERE created_at IS NOT NULL AND created_at <= ?",
            (cutoff,)
        )
        deleted = cursor.rowcount
        connection.commit()
    if deleted:
        log_msg(f"SQLite cleanup: deleted {deleted} setup record(s) older than 30 minutes.")
    return deleted


def claim_setup(setup_id, stale_after=120):
    """
    Atomically reserve a setup before Telegram send.
    A record is retained for at most 30 minutes from its first claim.
    """
    now = time.time()
    cutoff = now - MAX_SETUP_AGE_SECONDS
    with sqlite3.connect(ALERT_DB_PATH, timeout=10) as connection:
        connection.execute("BEGIN IMMEDIATE")
        row = connection.execute(
            "SELECT status, updated_at, created_at FROM alert_setups WHERE setup_id = ?",
            (setup_id,)
        ).fetchone()

        if row and row[2] is not None and row[2] <= cutoff:
            connection.execute("DELETE FROM alert_setups WHERE setup_id = ?", (setup_id,))
            row = None

        if row and row[0] == "sent":
            connection.commit()
            return False
        if row and row[0] == "claimed" and now - row[1] < stale_after:
            connection.commit()
            return False

        connection.execute("""
            INSERT INTO alert_setups (setup_id, status, updated_at, created_at)
            VALUES (?, 'claimed', ?, ?)
            ON CONFLICT(setup_id) DO UPDATE SET
                status = 'claimed', updated_at = excluded.updated_at
        """, (setup_id, now, now))
        connection.commit()
        return True


def update_setup_status(setup_id, status):
    with sqlite3.connect(ALERT_DB_PATH, timeout=10) as connection:
        connection.execute(
            "UPDATE alert_setups SET status = ?, updated_at = ? WHERE setup_id = ?",
            (status, time.time(), setup_id)
        )
        connection.commit()


def make_setup_id(symbol, setup, fvg):
    return "|".join((symbol, setup["alert_type"],
                     pd.Timestamp(setup["time"]).isoformat(),
                     pd.Timestamp(fvg["time"]).isoformat()))


def is_fvg_mitigated(df, fvg):
    later = df.loc[df.index > fvg["time"]]
    if later.empty:
        return False
    if fvg["type"] == "Bearish FVG":
        return bool((later["High"] >= fvg["bottom"]).any())
    return bool((later["Low"] <= fvg["top"]).any())

def evaluate_pair_ml_setup(symbol, setup, fvg, completed_3m, alert_close_time):
    """
    Computes all 27 features for the updated calibrated ensemble (XGB + LGBM + RF + ET)
    AND retains legacy 16-feature keys for backward compatibility with older .joblib bundles.
    Returns: (est_revisit_prob, raw_score, confidence_tier_label, feature_summary_dict)
    """
    crossing_time = setup["time"]
    fvg_time = fvg["time"]
    bundle = ml_bundles.get(symbol)
    defaults = PAIR_DEFAULTS.get(symbol, PAIR_DEFAULTS["EUR_JPY"])

    q25_gap = bundle.get("q25_gap", defaults["q25_gap"]) if bundle else defaults["q25_gap"]
    q50_gap = bundle.get("q50_gap", defaults["q50_gap"]) if bundle else defaults["q50_gap"]
    q75_gap = bundle.get("q75_gap", defaults["q75_gap"]) if bundle else defaults["q75_gap"]

    df_calc = completed_3m.copy()
    df_calc["SMA20"] = df_calc["Close"].rolling(M3_SMA_PERIOD).mean()
    df_calc["SMA50"] = df_calc["Close"].rolling(M5_SMA_PERIOD).mean()

    # 14-period True Range & ATR on M3 (matches training pipeline)
    prev_close = df_calc["Close"].shift(1)
    tr = pd.concat([
        df_calc["High"] - df_calc["Low"],
        (df_calc["High"] - prev_close).abs(),
        (df_calc["Low"] - prev_close).abs()
    ], axis=1).max(axis=1)
    df_calc["ATR14"] = tr.rolling(14).mean().bfill().replace(0, 1e-5)

    # 14-period RSI on M3 (matches training pipeline)
    delta = df_calc["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    df_calc["RSI14"] = (100.0 - (100.0 / (1.0 + gain / (loss + 1e-8)))).fillna(50.0)

    current = df_calc.iloc[-1]
    c_open = float(current["Open"])
    c_high = float(current["High"])
    c_low = float(current["Low"])
    c_close = float(current["Close"])
    cur_atr = float(current["ATR14"]) if pd.notna(current["ATR14"]) and current["ATR14"] > 0 else 1e-5
    cur_rsi = float(current["RSI14"]) if pd.notna(current["RSI14"]) else 50.0

    sma_3m_20 = float(current["SMA20"])
    sma_3m_50 = float(current["SMA50"]) if pd.notna(current["SMA50"]) else sma_3m_20

    sma_5m_50 = setup.get("m5_sma50")
    if sma_5m_50 is None or pd.isna(sma_5m_50):
        _, sma_5m_50 = get_m5_trend_and_sma(symbol)
    if sma_5m_50 is None or pd.isna(sma_5m_50):
        sma_5m_50 = sma_3m_50
    sma_5m_50 = float(sma_5m_50)

    trend_bullish = 1 if setup["trend"] == "bullish" else 0
    sign = 1.0 if trend_bullish == 1 else -1.0

    # Timing features
    fvg_to_cross_mins = (crossing_time - fvg_time).total_seconds() / 60.0
    wait_after_cross_mins = (alert_close_time - crossing_time).total_seconds() / 60.0
    total_setup_mins = (alert_close_time - fvg_time).total_seconds() / 60.0
    wait_to_total_ratio = wait_after_cross_mins / (abs(total_setup_mins) + 1.0)

    wait_rounded = round(wait_after_cross_mins, 1)
    is_precross_fvg = int(wait_rounded <= 3.0)
    is_sweetspot_wait = int(wait_rounded in [6.0, 9.0, 15.0, 18.0, 21.0, 24.0])
    is_peak_wait_9_or_18 = int(wait_rounded in [9.0, 18.0])
    is_late_wait_27_30 = int(wait_rounded in [27.0, 30.0])

    # Gap features
    fvg_top = float(fvg["top"])
    fvg_bottom = float(fvg["bottom"])
    gap_size = abs(fvg_top - fvg_bottom)
    is_optimal_gap = int((gap_size > q25_gap) and (gap_size <= q50_gap))
    is_exhaustion_gap = int(gap_size > q75_gap)
    gap_relative_to_median = gap_size / (q50_gap + 1e-8)

    # Cyclical & Session features (UTC)
    alert_utc = pd.Timestamp(alert_close_time)
    if alert_utc.tzinfo is not None:
        alert_utc = alert_utc.tz_convert("UTC")
    hour = int(alert_utc.hour)
    hour_float = hour + alert_utc.minute / 60.0
    hour_sin = float(np.sin(2.0 * np.pi * hour_float / 24.0))
    hour_cos = float(np.cos(2.0 * np.pi * hour_float / 24.0))
    is_london_open = int(7 <= hour < 10)
    is_ny_overlap = int(13 <= hour <= 16)
    is_late_session = int(hour >= 17)

    # Scale-invariant SMA spreads (basis points) + legacy raw diffs
    price_ref = abs(sma_3m_20) + 1e-8
    sma_3m_20_50_diff = (sma_3m_20 - sma_3m_50) * sign
    sma_3m20_5m50_diff = (sma_3m_20 - sma_5m_50) * sign
    sma_3m50_5m50_diff = (sma_3m_50 - sma_5m_50) * sign

    sma_3m_20_50_pct = sma_3m_20_50_diff / price_ref * 10000.0
    sma_3m20_5m50_pct = sma_3m20_5m50_diff / price_ref * 10000.0
    sma_3m50_5m50_pct = sma_3m50_5m50_diff / price_ref * 10000.0

    gap_to_3m_spread_ratio = gap_size / (abs(sma_3m_20 - sma_3m_50) + 1e-6)
    gap_to_5m_spread_ratio = gap_size / (abs(sma_3m_20 - sma_5m_50) + 1e-6)

    # ATR & Price-Action Geometry features
    if fvg["type"] == "Bearish FVG":
        dist_to_fvg = fvg_bottom - c_close
        rejection_wick = c_high - max(c_open, c_close)
        rsi_exhaustion = 100.0 - cur_rsi
    else:
        dist_to_fvg = c_close - fvg_top
        rejection_wick = min(c_open, c_close) - c_low
        rsi_exhaustion = cur_rsi

    dist_in_atr = dist_to_fvg / cur_atr
    gap_in_atr = gap_size / cur_atr
    wick_in_atr = rejection_wick / cur_atr
    sma20_prev3 = float(df_calc["SMA20"].iloc[-4]) if len(df_calc) >= 4 and pd.notna(df_calc["SMA20"].iloc[-4]) else sma_3m_20
    sma20_slope_atr = (sma_3m_20 - sma20_prev3) / cur_atr

    row_dict = {
        # --- Updated 27-Feature Model Inputs ---
        "Wait_After_Cross_Mins": wait_after_cross_mins,
        "FVG_to_Cross_Mins": fvg_to_cross_mins,
        "Total_Setup_Mins": total_setup_mins,
        "Wait_to_Total_Ratio": wait_to_total_ratio,
        "Is_PreCross_FVG": is_precross_fvg,
        "Is_SweetSpot_Wait": is_sweetspot_wait,
        "Is_Peak_Wait_9_or_18": is_peak_wait_9_or_18,
        "Is_Late_Wait_27_30": is_late_wait_27_30,
        "Gap_Relative_to_Median": gap_relative_to_median,
        "Is_Optimal_Gap": is_optimal_gap,
        "Is_Exhaustion_Gap": is_exhaustion_gap,
        "Hour_Sin": hour_sin,
        "Hour_Cos": hour_cos,
        "Is_London_Open": is_london_open,
        "Is_NY_Overlap": is_ny_overlap,
        "Is_Late_Session": is_late_session,
        "SMA_3m_20_50_Pct": sma_3m_20_50_pct,
        "SMA_3m20_5m50_Pct": sma_3m20_5m50_pct,
        "SMA_3m50_5m50_Pct": sma_3m50_5m50_pct,
        "Gap_to_3mSpread_Ratio": gap_to_3m_spread_ratio,
        "Gap_to_5mSpread_Ratio": gap_to_5m_spread_ratio,
        "Trend_Bullish": trend_bullish,
        "Dist_in_ATR": dist_in_atr,
        "Gap_in_ATR": gap_in_atr,
        "Wick_in_ATR": wick_in_atr,
        "SMA20_Slope_ATR": sma20_slope_atr,
        "RSI_Exhaustion": rsi_exhaustion,
        # --- Legacy 16-Feature Keys (Backward Compatibility) ---
        "FVG GAP": gap_size,
        "SMA_3m_20_50_Diff": sma_3m_20_50_diff,
        "SMA_3m20_5m50_Diff": sma_3m20_5m50_diff,
        "SMA_3m50_5m50_Diff": sma_3m50_5m50_diff,
    }

    if bundle is not None:
        try:
            feature_cols = bundle["features"]
            X_live = pd.DataFrame([row_dict])[feature_cols]

            p_xgb = float(bundle["xgb"].predict_proba(X_live)[0, 1])
            p_rf = float(bundle["rf"].predict_proba(X_live)[0, 1])
            et_model = bundle.get("et")
            lgbm_model = bundle.get("lgbm")

            if et_model is not None and lgbm_model is not None:
                p_et = float(et_model.predict_proba(X_live)[0, 1])
                p_lgbm = float(lgbm_model.predict_proba(X_live)[0, 1])
                raw_score = 0.40 * p_xgb + 0.30 * p_lgbm + 0.15 * p_rf + 0.15 * p_et
            elif et_model is not None:
                p_et = float(et_model.predict_proba(X_live)[0, 1])
                raw_score = 0.50 * p_xgb + 0.30 * p_rf + 0.20 * p_et
            else:
                raw_score = 0.55 * p_xgb + 0.45 * p_rf

            p30 = bundle.get("p30_cutoff", 0.45)
            p50 = bundle.get("p50_cutoff", 0.50)
            p70 = bundle.get("p70_cutoff", 0.55)
            p85 = bundle.get("p85_cutoff")
            tier_wrs = bundle.get("tier_winrates", {})

            if p85 is not None and raw_score >= p85:
                tier = "🚀 ULTRA CONVICTION (Top 15%)"
                tier_wr = tier_wrs.get("top15", tier_wrs.get("top30", 0.870))
            elif raw_score >= p70:
                tier = "🔥 HIGH CONVICTION (Top 30%)"
                tier_wr = tier_wrs.get("top30", 0.848)
            elif raw_score >= p50:
                tier = "✅ STRONG SETUP (Top 50%)"
                tier_wr = tier_wrs.get("top50", 0.817)
            elif raw_score >= p30:
                tier = "🟡 MODERATE SETUP (Top 70%)"
                tier_wr = tier_wrs.get("top70", 0.801)
            else:
                tier = "⚠️ LOW PROBABILITY (Bottom 30%)"
                tier_wr = tier_wrs.get("bottom30", 0.705)

            calibrator = bundle.get("calibrator")
            est_wr = float(calibrator.predict([raw_score])[0]) if calibrator is not None else float(tier_wr)

            return float(est_wr), raw_score, tier, row_dict
        except Exception as e:
            log_msg(f"ML prediction error for {symbol}, using empirical lookup: {e}")

    # Empirical fallback calibrated to JPY cross timing regimes
    wait_empirical_map = {
        3.0: 0.705, 6.0: 0.805, 9.0: 0.861, 12.0: 0.750,
        15.0: 0.832, 18.0: 0.860, 21.0: 0.804, 24.0: 0.826,
        27.0: 0.756, 30.0: 0.726, 33.0: 0.800
    }
    base_prob = wait_empirical_map.get(wait_rounded, 0.783)
    if is_exhaustion_gap:
        base_prob -= 0.028
    elif is_optimal_gap:
        base_prob += 0.015
    if is_ny_overlap:
        base_prob += 0.012
    elif is_late_session:
        base_prob -= 0.015

    base_prob = float(np.clip(base_prob, 0.50, 0.92))
    if base_prob >= 0.83:
        tier = "🔥 HIGH CONVICTION (Sweet-Spot Regime)"
    elif base_prob >= 0.79:
        tier = "✅ STRONG SETUP (Post-Cross Regime)"
    else:
        tier = "⚠️ LOWER PROBABILITY (Pre-Cross / Exhaustion)"

    return base_prob, base_prob, tier, row_dict

# ==============================
# ECONOMIC CALENDAR
# ==============================
def get_weekly_news():
    now = time.time()
    if news_cache["timestamp"] is None or now - news_cache["timestamp"] > 14400:
        url = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
        try:
            response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
            response.raise_for_status()
            news_cache["data"] = response.json()
            news_cache["timestamp"] = now
            log_msg("Economic calendar cache refreshed.")
        except Exception as e:
            log_msg(f"Calendar fetch error: {e}")
    return news_cache["data"]

def get_todays_pair_news(symbol, buffer_minutes=5):
    events = get_weekly_news()
    if not events:
        return []

    currencies = symbol.split("_")
    uk_tz = ZoneInfo("Europe/London")
    now_uk = datetime.now(uk_tz)
    today_date = now_uk.date()

    todays_events = []
    for event in events:
        country = event.get("country")
        impact = event.get("impact")

        if country in currencies and impact in ["High", "Medium"]:
            try:
                event_time_utc = pd.to_datetime(event["date"], utc=True)
                event_time_uk = event_time_utc.tz_convert(uk_tz)

                if event_time_uk.date() == today_date:
                    start_window = (event_time_uk - timedelta(minutes=buffer_minutes)).strftime("%H:%M")
                    end_window = (event_time_uk + timedelta(minutes=buffer_minutes)).strftime("%H:%M")
                    exact_time = event_time_uk.strftime("%H:%M")

                    diff_mins = (now_uk - event_time_uk).total_seconds() / 60.0
                    if abs(diff_mins) <= buffer_minutes:
                        status = "ACTIVE NOW"
                    elif diff_mins > buffer_minutes:
                        status = "Passed"
                    else:
                        status = "Upcoming"

                    todays_events.append({
                        "currency": country,
                        "title": event.get("title", "News Event"),
                        "impact": impact,
                        "time_uk": exact_time,
                        "window_uk": f"{start_window} - {end_window} UK",
                        "status": status,
                        "sort_key": event_time_uk.strftime("%H:%M:%S")
                    })
            except Exception:
                continue

    todays_events.sort(key=lambda x: x["sort_key"])
    for item in todays_events:
        item.pop("sort_key", None)
    return todays_events


# ==============================
# EXISTING SCANNER LOGIC WITH HEALTH CHECKS
# ==============================
def send_telegram(message):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        app_state["telegram_health"] = "Missing Credentials"
        log_msg("Telegram credentials are missing.")
        return False
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try:
        response = requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": message}, timeout=10)
        response.raise_for_status()
        if not response.json().get("ok"):
            app_state["telegram_health"] = "API Rejected"
            log_msg(f"Telegram rejected message: {response.json()}")
            return False
        app_state["telegram_health"] = "Healthy 🟢"
        return True
    except requests.RequestException as e:
        app_state["telegram_health"] = "Error 🔴"
        log_msg(f"Telegram error: {e}")
        return False

def get_oanda_candles(granularity="M3", count=60, symbol=None):
    """Fetch candles for an explicit pair; independent of dashboard selection."""
    symbol = symbol or app_state["symbol"]
    if symbol not in AVAILABLE_PAIRS:
        log_msg(f"Rejected unsupported OANDA instrument: {symbol}")
        return pd.DataFrame()
    if not OANDA_TOKEN:
        pair_status[symbol]["oanda_health"] = "Missing Credentials"
        app_state["oanda_health"] = "Missing Credentials"
        log_msg("OANDA token is missing. Set OANDA_TOKEN.")
        return pd.DataFrame()

    url = f"{OANDA_URL}/v3/instruments/{symbol}/candles"
    headers = {"Authorization": f"Bearer {OANDA_TOKEN}",
               "Accept-Datetime-Format": "RFC3339"}
    params = {"granularity": granularity, "count": count, "price": "M"}
    try:
        response = requests.get(url, headers=headers, params=params, timeout=10)
        response.raise_for_status()
        candles = []
        for candle in response.json().get("candles", []):
            mid = candle.get("mid")
            if not mid:
                continue
            candles.append({
                "Time": pd.to_datetime(candle["time"], utc=True),
                "Open": float(mid["o"]), "High": float(mid["h"]),
                "Low": float(mid["l"]), "Close": float(mid["c"]),
                "Complete": bool(candle.get("complete", False))
            })
        if not candles:
            log_msg(f"[{symbol}] OANDA returned no usable {granularity} candles.")
            return pd.DataFrame()
        pair_status[symbol]["oanda_health"] = "Healthy 🟢"
        return pd.DataFrame(candles).set_index("Time").sort_index()
    except requests.HTTPError as exc:
        response = exc.response
        status = response.status_code if response is not None else "unknown"
        body = response.text[:1000] if response is not None else "No response body"
        pair_status[symbol]["oanda_health"] = f"HTTP Error {status} 🔴"
        log_msg(f"OANDA HTTP error | {symbol} {granularity} | status={status} | body={body}")
    except requests.Timeout as exc:
        pair_status[symbol]["oanda_health"] = "Timeout 🔴"
        log_msg(f"OANDA timeout | {symbol} {granularity} | {exc}")
    except requests.ConnectionError as exc:
        pair_status[symbol]["oanda_health"] = "Connection Error 🔴"
        log_msg(f"OANDA connection error | {symbol} {granularity} | {exc}")
    except requests.RequestException as exc:
        pair_status[symbol]["oanda_health"] = "Request Error 🔴"
        log_msg(f"OANDA request error | {symbol} {granularity} | {exc}")
    except (ValueError, KeyError, TypeError) as exc:
        pair_status[symbol]["oanda_health"] = "Data Error 🔴"
        log_msg(f"OANDA parsing error | {symbol} {granularity} | {type(exc).__name__}: {exc}")
    return pd.DataFrame()

def get_m5_trend_and_sma(symbol, force_refresh=False):
    """Cache M5 trend separately for each pair."""
    cached = m5_cache[symbol]
    now = time.monotonic()
    if not force_refresh and now - cached["timestamp"] < M5_CACHE_SECONDS:
        return cached["trend"], cached["sma50"]
    df = get_oanda_candles("M5", count=100, symbol=symbol)
    if df.empty:
        return cached["trend"], cached["sma50"]
    df = df[df["Complete"]].copy()
    if len(df) < M5_SMA_PERIOD:
        return None, None
    sma50 = float(df["Close"].rolling(M5_SMA_PERIOD).mean().iloc[-1])
    close = float(df["Close"].iloc[-1])
    trend = "bullish" if close > sma50 else "bearish" if close < sma50 else None
    cached.update({"timestamp": now, "trend": trend, "sma50": sma50})
    pair_status[symbol]["m5_trend"] = trend.upper() if trend else "Neutral"
    return trend, sma50

def detect_sma20_cross(df):
    df = df[df["Complete"]].copy()
    if len(df) < M3_SMA_PERIOD + 1:
        return None
    df["SMA20"] = df["Close"].rolling(M3_SMA_PERIOD).mean()
    previous, current = df.iloc[-2], df.iloc[-1]
    if previous["Close"] <= previous["SMA20"] and current["Close"] > current["SMA20"]:
        return "above"
    if previous["Close"] >= previous["SMA20"] and current["Close"] < current["SMA20"]:
        return "below"
    return None

def find_matching_fvg(df, direction, crossing_time, lookback=10, lookahead=10):
    """
    Find the earliest fresh FVG across pre-cross, at-cross and post-cross candles.
    All three FVG-forming candles must have the required candle colour.
    Mitigated candidates are skipped so another qualifying FVG can be considered.
    """
    df = df[df["Complete"]].copy().sort_index()
    if len(df) < 3 or crossing_time not in df.index:
        return None
    crossing_pos = df.index.get_loc(crossing_time)
    first_i = max(2, crossing_pos - lookback)
    last_i = min(len(df) - 1, crossing_pos + lookahead)

    for i in range(first_i, last_i + 1):
        first, second, third = df.iloc[i - 2], df.iloc[i - 1], df.iloc[i]
        if direction == "bearish":
            candles_match = all(c["Close"] < c["Open"] for c in (first, second, third))
            gap_exists = first["Low"] > third["High"]
            if not (candles_match and gap_exists):
                continue
            candidate = {"type": "Bearish FVG", "time": df.index[i],
                         "top": float(first["Low"]), "bottom": float(third["High"])}
        elif direction == "bullish":
            candles_match = all(c["Close"] > c["Open"] for c in (first, second, third))
            gap_exists = first["High"] < third["Low"]
            if not (candles_match and gap_exists):
                continue
            candidate = {"type": "Bullish FVG", "time": df.index[i],
                         "top": float(third["Low"]), "bottom": float(first["High"])}
        else:
            raise ValueError("direction must be 'bullish' or 'bearish'")

        if not is_fvg_mitigated(df, candidate):
            return candidate
    return None

def is_trading_time():
    uk_time = datetime.now(ZoneInfo("Europe/London"))
    return 7 <= uk_time.hour < 20

def process_pair(symbol):
    """Run one independent scan cycle for EUR_JPY or GBP_JPY."""
    pair_status[symbol]["scanner_status"] = "Fetching M3 candles..."
    df_3m = get_oanda_candles("M3", count=80, symbol=symbol)
    if df_3m.empty:
        pair_status[symbol]["scanner_status"] = "Waiting for OANDA data"
        return

    completed = df_3m[df_3m["Complete"]].copy().sort_index()
    if len(completed) < max(M3_SMA_PERIOD + 1, M5_SMA_PERIOD):
        pair_status[symbol]["scanner_status"] = "Waiting for enough completed candles"
        return

    latest_time = completed.index[-1]
    pair_status[symbol]["last_candle"] = latest_time.isoformat()
    latest_close = latest_time + pd.Timedelta(minutes=3)
    candle_age = (pd.Timestamp.now(tz="UTC") - latest_close).total_seconds()
    if candle_age > 6 * 60:
        pair_status[symbol]["scanner_status"] = "Latest completed candle is stale"
        return

    pair_status[symbol]["scanner_status"] = "Checking M3 SMA20 crossing..."
    crossing = detect_sma20_cross(completed)
    if crossing is not None and latest_time != last_processed_crossing[symbol]:
        last_processed_crossing[symbol] = latest_time
        trend, m5_sma50 = get_m5_trend_and_sma(symbol, force_refresh=True)

        # Counter-trend pullback: bullish M5 + M3 cross below = LONG with bearish FVG.
        # Bearish M5 + M3 cross above = SHORT with bullish FVG.
        if trend == "bullish" and crossing == "below":
            alert_type, label, fvg_direction = "long", "LONG PULLBACK", "bearish"
        elif trend == "bearish" and crossing == "above":
            alert_type, label, fvg_direction = "short", "SHORT PULLBACK", "bullish"
        else:
            alert_type = None

        if alert_type:
            pending_crossings[symbol][latest_time.isoformat()] = {
                "symbol": symbol, "time": latest_time, "created_at": time.time(), "crossing": crossing,
                "trend": trend, "m5_sma50": m5_sma50,
                "alert_type": alert_type, "alert_label": label,
                "fvg_direction": fvg_direction,
                "price": float(completed.iloc[-1]["Close"])
            }
            log_msg(f"[{symbol}] {crossing.upper()} SMA20 cross | M5={trend} | waiting for {fvg_direction} FVG.")

    pair_status[symbol]["scanner_status"] = f"Processing {len(pending_crossings[symbol])} pending setup(s)"
    for key, setup in list(pending_crossings[symbol].items()):
        crossing_time = setup["time"]
        setup_created_at = setup.get("created_at")
        if setup_created_at is None:
            # Backward-compatible fallback for any in-memory setup created before this update.
            setup_created_at = pd.Timestamp(crossing_time).timestamp()
        if time.time() - float(setup_created_at) >= MAX_SETUP_AGE_SECONDS:
            pending_crossings[symbol].pop(key, None)
            log_msg(f"[{symbol}] Pending setup expired after 30 minutes; it will not alert.")
            continue
        if crossing_time not in completed.index:
            pending_crossings[symbol].pop(key, None)
            continue
        crossing_pos = completed.index.get_loc(crossing_time)
        age = (len(completed) - 1) - crossing_pos
        if age > FVG_LOOKAHEAD:
            pending_crossings[symbol].pop(key, None)
            log_msg(f"[{symbol}] Setup expired; no matching FVG within {FVG_LOOKAHEAD} candles.")
            continue

        fvg = find_matching_fvg(completed, setup["fvg_direction"], crossing_time,
                                FVG_LOOKBACK, FVG_LOOKAHEAD)
        if fvg is None:
            continue

        setup_id = make_setup_id(symbol, setup, fvg)
        with alert_lock:
            if setup_id in sending_setups:
                pending_crossings[symbol].pop(key, None)
                continue
            sending_setups.add(setup_id)

        try:
            claimed = claim_setup(setup_id)
        except sqlite3.Error as exc:
            with alert_lock:
                sending_setups.discard(setup_id)
            log_msg(f"[{symbol}] Could not claim alert safely (SQLite): {exc}")
            continue

        if not claimed:
            with alert_lock:
                sending_setups.discard(setup_id)
            pending_crossings[symbol].pop(key, None)
            continue

        try:
            # Only completed M3 candles are passed into signal evaluation.
            alert_close_time = latest_time + pd.Timedelta(minutes=3)
            probability, raw_score, tier, features = evaluate_pair_ml_setup(
                symbol, setup, fvg, completed, alert_close_time
            )
            wait_mins = features["Wait_After_Cross_Mins"]
            gap_size = features["FVG GAP"]
            regime = "Post-Cross FVG" if fvg["time"] > crossing_time else "Pre/At-Cross FVG"
            fvg_uk = fvg["time"].tz_convert("Europe/London")
            now_uk = pd.Timestamp.now(tz="Europe/London")
            delay_seconds = max(0.0, (now_uk - fvg_uk).total_seconds())

            message = (
                f"{symbol} - {setup['alert_label']}\n"
                f"• ML 60m Revisit Est: {probability:.1%}\n"
                f"• Signal Tier: {tier}\\n"
                # f"• Setup Regime: {regime}\\n"
                # f"• Wait After Cross: {wait_mins:.0f} mins\\n"
                f"• Trend (M5 SMA50): {setup['trend'].upper()}\n"
                # f"• M3 crossing: {setup['crossing'].upper()} SMA20\n"
                f"• FVG: {fvg['type']} ({fvg['bottom']:.3f} - {fvg['top']:.3f})\n"
                # f"• FVG Gap: {gap_size:.3f}\n"
                # f"• Time since FVG candle: {delay_seconds:.0f}s"
            )

            if send_telegram(message):
                update_setup_status(setup_id, "sent")
                pending_crossings[symbol].pop(key, None)
                pair_status[symbol]["last_alert"] = datetime.now(
                    ZoneInfo("Europe/London")).strftime("%Y-%m-%d %H:%M:%S UK")
                log_msg(f"[{symbol}] Telegram alert sent once | {fvg['type']} | score={raw_score:.3f}")
            else:
                update_setup_status(setup_id, "failed")
                log_msg(f"[{symbol}] Telegram send failed; setup remains pending for retry.")
        finally:
            with alert_lock:
                sending_setups.discard(setup_id)

    pair_status[symbol]["scanner_status"] = "Idle — monitoring"


def run_fvg_scanner():
    initialise_alert_db()
    load_ml_models()
    log_msg("Scanner started: EUR_JPY and GBP_JPY are both enabled.")
    last_db_cleanup = 0.0

    while True:
        cycle_start = time.monotonic()
        if cycle_start - last_db_cleanup >= DB_CLEANUP_INTERVAL_SECONDS:
            try:
                cleanup_expired_setup_records()
            except sqlite3.Error as exc:
                log_msg(f"SQLite cleanup error: {exc}")
            last_db_cleanup = cycle_start
        if not is_trading_time():
            for pair in AVAILABLE_PAIRS:
                pair_status[pair]["scanner_status"] = "Paused (outside UK 07:00–20:00)"
        else:
            for pair in AVAILABLE_PAIRS:
                try:
                    process_pair(pair)
                except Exception as exc:
                    pair_status[pair]["scanner_status"] = "Scanner error"
                    log_msg(f"[{pair}] Scanner error: {type(exc).__name__}: {exc}")

        elapsed = time.monotonic() - cycle_start
        time.sleep(max(1, SCAN_INTERVAL - elapsed))


HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta name="theme-color" content="#080d18">
    <title>FX Pulse | Trading Intelligence</title>

    <style>
        :root {
            --bg: #080d18;
            --bg-secondary: #0c1424;
            --panel: rgba(16, 27, 45, 0.88);
            --panel-hover: rgba(20, 35, 57, 0.96);
            --cyan: #00e5ff;
            --blue: #3984ff;
            --purple: #a855f7;
            --green: #00f5a0;
            --red: #ff4d6d;
            --yellow: #ffbd4a;
            --text: #f1f5f9;
            --muted: #8493a8;
            --border: rgba(148, 163, 184, 0.13);
            --log-bg: #050a13;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        html {
            scroll-behavior: smooth;
        }

        body {
            min-height: 100vh;
            font-family: Inter, -apple-system, BlinkMacSystemFont,
                'Segoe UI', Roboto, sans-serif;
            background:
                radial-gradient(ellipse at 10% 10%,
                    rgba(0, 229, 255, .08), transparent 38%),
                radial-gradient(ellipse at 90% 85%,
                    rgba(168, 85, 247, .09), transparent 38%),
                var(--bg);
            background-attachment: fixed;
            color: var(--text);
            padding: 26px;
            overflow-x: hidden;
        }

        /* Background ambient effects */

        .fx-glow {
            position: fixed;
            width: 300px;
            height: 300px;
            border-radius: 50%;
            filter: blur(110px);
            opacity: .15;
            pointer-events: none;
            z-index: -1;
            animation: floatGlow 14s ease-in-out infinite alternate;
        }

        .fx-glow.cyan {
            background: var(--cyan);
            top: 8%;
            left: -130px;
        }

        .fx-glow.purple {
            background: var(--purple);
            right: -130px;
            bottom: 5%;
            animation-delay: -7s;
        }

        @keyframes floatGlow {
            0% {
                transform: translate(0, 0) scale(1);
            }
            50% {
                transform: translate(70px, 50px) scale(1.2);
            }
            100% {
                transform: translate(-20px, 100px) scale(.9);
            }
        }

        /* Main layout */

        .dashboard {
            max-width: 1500px;
            margin: auto;
            position: relative;
            z-index: 1;
        }

        .header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 20px;
            margin-bottom: 28px;
            flex-wrap: wrap;
        }

        .brand {
            display: flex;
            align-items: center;
            gap: 15px;
        }

        .brand-icon {
            width: 48px;
            height: 48px;
            display: grid;
            place-items: center;
            border-radius: 14px;
            background: linear-gradient(135deg, #00c6e8, #6654ed);
            color: white;
            font-size: 23px;
            font-weight: 900;
            box-shadow: 0 0 25px rgba(0, 229, 255, .2);
        }

        .brand-name {
            font-size: 1.65rem;
            font-weight: 850;
            letter-spacing: 2px;
            background: linear-gradient(
                90deg,
                #fff,
                var(--cyan),
                #fff,
                var(--purple),
                #fff
            );
            background-size: 300% auto;
            color: transparent;
            background-clip: text;
            -webkit-background-clip: text;
            animation: logoShine 6s linear infinite;
        }

        @keyframes logoShine {
            to {
                background-position: 300% center;
            }
        }

        .brand-subtitle {
            color: var(--muted);
            font-size: .75rem;
            letter-spacing: 2px;
            margin-top: 3px;
        }

        .header-right {
            display: flex;
            align-items: center;
            gap: 16px;
            flex-wrap: wrap;
        }

        .live-indicator {
            display: flex;
            align-items: center;
            gap: 9px;
            padding: 10px 15px;
            border: 1px solid rgba(0, 245, 160, .2);
            background: rgba(0, 245, 160, .05);
            border-radius: 30px;
            color: var(--green);
            font-size: .75rem;
            font-weight: 700;
            letter-spacing: 1px;
        }

        .status-dot {
            width: 9px;
            height: 9px;
            border-radius: 50%;
            background: var(--green);
            box-shadow: 0 0 10px var(--green);
            animation: statusPulse 1.5s ease-in-out infinite;
            display: inline-block;
            flex-shrink: 0;
        }

        .status-dot.warning {
            background: var(--yellow);
            box-shadow: 0 0 10px var(--yellow);
        }

        .status-dot.offline {
            background: var(--red);
            box-shadow: 0 0 10px var(--red);
            animation: none;
        }

        @keyframes statusPulse {
            0%, 100% {
                opacity: 1;
                transform: scale(1);
            }
            50% {
                opacity: .45;
                transform: scale(1.4);
            }
        }

        .clock {
            color: var(--muted);
            font-size: .85rem;
            font-variant-numeric: tabular-nums;
        }

        /* Main grid */

        .main-grid {
            display: grid;
            grid-template-columns: minmax(320px, .9fr) minmax(0, 1.6fr);
            gap: 22px;
            align-items: stretch;
        }

        .left-column {
            display: flex;
            flex-direction: column;
            gap: 20px;
            min-width: 0;
        }

        .right-column {
            min-width: 0;
        }

        .panel {
            background: var(--panel);
            border: 1px solid var(--border);
            border-radius: 17px;
            padding: 22px;
            backdrop-filter: blur(16px);
            box-shadow: 0 10px 35px rgba(0, 0, 0, .2);
            transition:
                transform .3s ease,
                border-color .3s ease,
                box-shadow .3s ease;
            min-width: 0;
        }

        .panel:hover {
            border-color: rgba(0, 229, 255, .28);
            box-shadow: 0 12px 38px rgba(0, 229, 255, .06);
        }

        .panel-heading {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 12px;
            margin-bottom: 18px;
        }

        .panel-title {
            font-size: .88rem;
            font-weight: 750;
            letter-spacing: 1.3px;
            text-transform: uppercase;
            color: #e2e8f0;
        }

        .panel-tag {
            color: var(--cyan);
            font-size: .65rem;
            letter-spacing: 1px;
            font-weight: 700;
            background: rgba(0, 229, 255, .08);
            padding: 5px 9px;
            border-radius: 6px;
        }

        .section-divider {
            border: 0;
            border-top: 1px solid var(--border);
            margin: 22px 0;
        }

        /* Pair selector */

        .control-group {
            display: flex;
            gap: 10px;
        }

        select {
            flex: 1;
            min-width: 0;
            padding: 12px 14px;
            background: #080f1d;
            color: var(--text);
            border: 1px solid var(--border);
            border-radius: 9px;
            font-size: .9rem;
            outline: none;
            cursor: pointer;
        }

        select:focus {
            border-color: var(--cyan);
            box-shadow: 0 0 0 3px rgba(0, 229, 255, .08);
        }

        button {
            padding: 12px 20px;
            border: 0;
            border-radius: 9px;
            background: linear-gradient(135deg, #00bcd4, #3975ee);
            color: white;
            font-size: .85rem;
            font-weight: 750;
            cursor: pointer;
            transition: all .25s ease;
            white-space: nowrap;
        }

        button:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 20px rgba(0, 229, 255, .2);
        }

        button:active {
            transform: scale(.97);
        }

        /* Scanner activity waveform */

        .activity-box {
            background: rgba(5, 10, 19, .6);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 14px;
            margin-top: 17px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 12px;
        }

        .activity-label {
            color: var(--muted);
            font-size: .72rem;
            letter-spacing: 1px;
        }

        .activity-value {
            color: var(--green);
            font-size: .78rem;
            font-weight: 750;
            margin-top: 5px;
        }

        .market-wave {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 4px;
            height: 45px;
            flex-shrink: 0;
        }

        .market-wave span {
            display: block;
            width: 4px;
            height: 12px;
            border-radius: 8px;
            background: linear-gradient(
                180deg,
                var(--cyan),
                var(--purple)
            );
            box-shadow: 0 0 8px rgba(0, 229, 255, .35);
            animation: waveMove .9s ease-in-out infinite alternate;
        }

        .market-wave span:nth-child(2n) {
            animation-delay: -.3s;
        }

        .market-wave span:nth-child(3n) {
            animation-delay: -.6s;
        }

        .market-wave span:nth-child(4n) {
            animation-delay: -.45s;
        }

        @keyframes waveMove {
            from {
                height: 7px;
                opacity: .5;
            }
            to {
                height: 40px;
                opacity: 1;
            }
        }

        /* Pair status cards */

        #pair-status-container {
            display: grid;
            grid-template-columns: 1fr;
            gap: 10px;
        }

        .pair-card {
            background: rgba(5, 12, 24, .72);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 15px;
            transition: all .25s ease;
            animation: cardEnter .5s ease both;
        }

        .pair-card:hover {
            border-color: rgba(0, 229, 255, .35);
            transform: translateY(-3px);
            background: var(--panel-hover);
        }

        .pair-card-top {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 10px;
            margin-bottom: 10px;
        }

        .pair-name {
            font-size: 1rem;
            font-weight: 800;
            letter-spacing: .7px;
        }

        .pair-health {
            font-size: .72rem;
            font-weight: 700;
        }

        .pair-info {
            color: var(--muted);
            font-size: .78rem;
            line-height: 1.8;
        }

        .pair-last-alert {
            margin-top: 8px;
            padding-top: 8px;
            border-top: 1px solid var(--border);
            color: #b8c6d8;
            font-size: .72rem;
        }

        @keyframes cardEnter {
            from {
                opacity: 0;
                transform: translateY(8px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        /* Health status rows */

        .status-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 12px;
            padding: 12px 0;
            border-bottom: 1px solid var(--border);
        }

        .status-row:last-child {
            border-bottom: none;
            padding-bottom: 0;
        }

        .status-label {
            color: var(--muted);
            font-size: .82rem;
        }

        .status-val {
            font-size: .82rem;
            font-weight: 750;
            text-align: right;
            overflow-wrap: anywhere;
        }

        .status-value-wrap {
            display: flex;
            align-items: center;
            gap: 8px;
            text-align: right;
        }

        /* Scanner task */

        .task-value {
            color: var(--yellow);
        }

        /* News cards */

        #news-container {
            display: flex;
            flex-direction: column;
            gap: 9px;
        }

        .news-card {
            background: rgba(5, 12, 24, .72);
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 13px;
            transition: border-color .2s ease;
        }

        .news-card:hover {
            border-color: rgba(0, 229, 255, .3);
        }

        .news-top {
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 8px;
            margin-bottom: 8px;
            flex-wrap: wrap;
        }

        .news-impact {
            font-size: .7rem;
            font-weight: 800;
            letter-spacing: .7px;
        }

        .news-status {
            font-size: .7rem;
            font-weight: 700;
        }

        .news-title {
            font-size: .83rem;
            font-weight: 650;
            margin-bottom: 8px;
        }

        .news-meta {
            display: flex;
            justify-content: space-between;
            gap: 8px;
            flex-wrap: wrap;
            color: var(--muted);
            font-size: .72rem;
        }

        .empty-news {
            color: var(--green);
            font-size: .82rem;
            padding: 12px 0;
        }

        /* Logs panel */

        .logs-panel {
            height: 100%;
            display: flex;
            flex-direction: column;
            min-height: 600px;
        }

        .logs-heading-right {
            display: flex;
            align-items: center;
            gap: 9px;
        }

        .terminal-label {
            color: var(--green);
            font-size: .67rem;
            font-weight: 750;
            letter-spacing: 1px;
        }

        #log-container {
            position: relative;
            overflow: auto;
            flex: 1;
            min-height: 450px;
            max-height: 75vh;
            padding: 18px;
            border: 1px solid rgba(0, 229, 255, .12);
            border-radius: 12px;
            background: var(--log-bg);
            color: #a3e6b5;
            font-family: 'Consolas', 'Monaco', monospace;
            font-size: .79rem;
            line-height: 1.9;
            white-space: pre-wrap;
            overflow-wrap: anywhere;
            box-shadow: inset 0 0 30px rgba(0, 0, 0, .35);
        }

        /* Scanner sweep over logs */

        .logs-panel #log-container::after {
            content: "";
            position: absolute;
            top: -100%;
            left: 0;
            width: 100%;
            height: 75%;
            pointer-events: none;
            background: linear-gradient(
                180deg,
                transparent,
                rgba(0, 229, 255, .055),
                transparent
            );
            animation: scannerSweep 6s linear infinite;
        }

        @keyframes scannerSweep {
            0% {
                top: -100%;
            }
            100% {
                top: 150%;
            }
        }

        /* Scrollbars */

        ::-webkit-scrollbar {
            width: 7px;
            height: 7px;
        }

        ::-webkit-scrollbar-track {
            background: #080f1d;
            border-radius: 5px;
        }

        ::-webkit-scrollbar-thumb {
            background: #263c55;
            border-radius: 5px;
        }

        ::-webkit-scrollbar-thumb:hover {
            background: var(--cyan);
        }

        /* Footer */

        .footer {
            display: flex;
            justify-content: space-between;
            gap: 12px;
            flex-wrap: wrap;
            padding: 20px 4px 5px;
            color: #65758b;
            font-size: .7rem;
            letter-spacing: .5px;
        }

        /* Responsive layout */

        @media (max-width: 1050px) {
            .main-grid {
                grid-template-columns: 1fr;
            }

            .logs-panel {
                min-height: 480px;
            }

            #log-container {
                min-height: 380px;
                max-height: 55vh;
            }
        }

        @media (max-width: 600px) {
            body {
                padding: 15px;
            }

            .header {
                align-items: flex-start;
                margin-bottom: 20px;
            }

            .brand-name {
                font-size: 1.35rem;
            }

            .brand-icon {
                width: 42px;
                height: 42px;
                font-size: 20px;
            }

            .header-right {
                width: 100%;
                justify-content: space-between;
            }

            .panel {
                padding: 17px;
                border-radius: 14px;
            }

            .main-grid {
                gap: 16px;
            }

            .left-column {
                gap: 16px;
            }

            .control-group {
                flex-direction: column;
            }

            .control-group button {
                width: 100%;
            }

            .status-row {
                align-items: flex-start;
            }

            .status-val {
                max-width: 58%;
            }

            .logs-panel {
                min-height: 430px;
            }

            #log-container {
                min-height: 330px;
                max-height: 55vh;
                font-size: .72rem;
                padding: 13px;
            }

            .footer {
                flex-direction: column;
            }
        }

        /* Accessibility */

        @media (prefers-reduced-motion: reduce) {
            *,
            *::before,
            *::after {
                animation-duration: .01ms !important;
                animation-iteration-count: 1 !important;
                transition-duration: .01ms !important;
                scroll-behavior: auto !important;
            }
        }
    </style>
</head>

<body>

    <div class="fx-glow cyan"></div>
    <div class="fx-glow purple"></div>

    <div class="dashboard">

        <!-- HEADER -->

        <header class="header">

            <div class="brand">
                <div class="brand-icon">FX</div>

                <div>
                    <div class="brand-name">FX PULSE</div>
                    <div class="brand-subtitle">
                        INTELLIGENT TRADING ALERT SYSTEM
                    </div>
                </div>
            </div>

            <div class="header-right">

                <div class="live-indicator">
                    <span class="status-dot"></span>
                    SYSTEM MONITOR
                </div>

                <div class="clock" id="uk-clock">
                    UK TIME --:--:--
                </div>

            </div>

        </header>


        <!-- MAIN DASHBOARD -->

        <main class="main-grid">

            <!-- LEFT COLUMN -->

            <div class="left-column">

                <!-- SCANNER CONTROL PANEL -->

                <section class="panel">

                    <div class="panel-heading">
                        <h2 class="panel-title">Dashboard View</h2>
                        <span class="panel-tag">DUAL PAIR</span>
                    </div>

                    <p style="color:var(--muted);font-size:.8rem;margin-bottom:15px;">
                        Both currency pairs are scanned independently.
                        Select a pair to change the dashboard view.
                    </p>

                    <div class="control-group">

                        <select id="symbol-select">
                            {% for pair in pairs %}
                            <option value="{{ pair }}"
                                {% if pair == current_symbol %}selected{% endif %}>
                                {{ pair }}
                            </option>
                            {% endfor %}
                        </select>

                        <button id="apply-button" onclick="updateSymbol()">
                            Apply
                        </button>

                    </div>

                    <div class="activity-box">

                        <div>
                            <div class="activity-label">SCANNER ACTIVITY</div>
                            <div class="activity-value">
                                MONITORING MARKETS
                            </div>
                        </div>

                        <div class="market-wave" aria-label="Animated visual activity indicator">
                            <span></span><span></span><span></span><span></span>
                            <span></span><span></span><span></span><span></span>
                            <span></span><span></span><span></span><span></span>
                            <span></span><span></span><span></span><span></span>
                        </div>

                    </div>

                    <hr class="section-divider">

                    <!-- BOTH PAIR SCANNERS -->

                    <div class="panel-heading">
                        <h2 class="panel-title">Pair Scanners</h2>
                        <span class="panel-tag">LIVE STATUS</span>
                    </div>

                    <div id="pair-status-container">
                        <div style="color:var(--muted);font-size:.8rem;">
                            Loading pair status...
                        </div>
                    </div>

                </section>


                <!-- API AND ML HEALTH -->

                <section class="panel">

                    <div class="panel-heading">
                        <h2 class="panel-title">API & ML Health</h2>
                        <span class="panel-tag">SELECTED PAIR</span>
                    </div>

                    <div class="status-row">
                        <span class="status-label">OANDA API</span>
                        <div class="status-value-wrap">
                            <span class="status-dot" id="oanda-dot"></span>
                            <span class="status-val" id="oanda-health">...</span>
                        </div>
                    </div>

                    <div class="status-row">
                        <span class="status-label">Telegram API</span>
                        <div class="status-value-wrap">
                            <span class="status-dot" id="tele-dot"></span>
                            <span class="status-val" id="tele-health">...</span>
                        </div>
                    </div>

                    <div class="status-row">
                        <span class="status-label">ML Revisit Engine</span>
                        <div class="status-value-wrap">
                            <span class="status-dot" id="ml-dot"></span>
                            <span class="status-val" id="ml-health">...</span>
                        </div>
                    </div>

                    <hr class="section-divider">

                    <div class="panel-heading">
                        <h2 class="panel-title">Scanner Activity</h2>
                        <span class="panel-tag">ENGINE</span>
                    </div>

                    <div class="status-row">
                        <span class="status-label">Current Task</span>
                        <span class="status-val task-value" id="scan-task">...</span>
                    </div>

                    <div class="status-row">
                        <span class="status-label">M5 Trend Filter</span>
                        <span class="status-val" id="m5-trend">...</span>
                    </div>

                </section>


                <!-- ECONOMIC CALENDAR -->

                <section class="panel">

                    <div class="panel-heading">
                        <h2 class="panel-title">Economic Calendar</h2>
                        <span class="panel-tag">UK TIME</span>
                    </div>

                    <div id="news-container">
                        <div style="color:var(--muted);font-size:.8rem;">
                            Checking calendar...
                        </div>
                    </div>

                </section>

            </div>


            <!-- RIGHT COLUMN: SYSTEM LOGS -->

            <div class="right-column">

                <section class="panel logs-panel">

                    <div class="panel-heading">

                        <div>
                            <h2 class="panel-title">System Logs</h2>
                            <p style="color:var(--muted);font-size:.75rem;margin-top:5px;">
                                Real-time scanner activity and system events
                            </p>
                        </div>

                        <div class="logs-heading-right">
                            <span class="status-dot"></span>
                            <span class="terminal-label">LIVE TERMINAL</span>
                        </div>

                    </div>

                    <div id="log-container">
                        Loading logs...
                    </div>

                </section>

            </div>

        </main>


        <!-- FOOTER -->

        <footer class="footer">
            <span>FX PULSE · TRADING INTELLIGENCE SYSTEM</span>
            <span>EUR/JPY & GBP/JPY MONITORING</span>
        </footer>

    </div>


    <!-- JAVASCRIPT -->

    <script>

        // UK CLOCK

        function updateClock() {
            const now = new Date();

            const time = now.toLocaleTimeString('en-GB', {
                timeZone: 'Europe/London',
                hour: '2-digit',
                minute: '2-digit',
                second: '2-digit'
            });

            document.getElementById('uk-clock').innerText =
                'UK TIME ' + time;
        }

        updateClock();
        setInterval(updateClock, 1000);


        // UPDATE SELECTED PAIR

        function updateSymbol() {

            const newSymbol =
                document.getElementById("symbol-select").value;

            const btn = document.getElementById("apply-button");
            const originalText = btn.innerText;

            btn.innerText = "Applying...";
            btn.disabled = true;

            fetch('/api/settings', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    symbol: newSymbol
                })
            })

            .then(res => {
                if (!res.ok) {
                    throw new Error("Unable to update symbol");
                }
                return res.json();
            })

            .then(data => {

                btn.innerText = "Updated";

                const taskRow = document.getElementById('scan-task');

                taskRow.style.color = '#00f5a0';

                setTimeout(() => {
                    taskRow.style.color = '';
                    btn.innerText = originalText;
                    btn.disabled = false;
                }, 1500);

                fetchData();

            })

            .catch(error => {

                console.error("Symbol update error:", error);

                btn.innerText = "Error";

                setTimeout(() => {
                    btn.innerText = originalText;
                    btn.disabled = false;
                }, 2000);

            });

        }


        // RENDER ECONOMIC NEWS

        function renderNews(newsList, symbol) {

            const container =
                document.getElementById('news-container');

            if (!newsList || newsList.length === 0) {

                container.innerHTML = `
                    <div class="empty-news">
                        🟢 No High/Medium news today for ${symbol}
                    </div>
                `;

                return;
            }

            container.innerHTML = newsList.map(item => {

                const badgeColor =
                    item.impact === 'High' ? '#ff4d6d' : '#ffbd4a';

                const statusColor =
                    item.status === 'ACTIVE NOW'
                        ? '#ff4d6d'
                        : (item.status === 'Passed'
                            ? '#8493a8'
                            : '#ffbd4a');

                return `

                    <div class="news-card">

                        <div class="news-top">

                            <span class="news-impact"
                                style="color:${badgeColor};">
                                ${item.currency} • ${item.impact.toUpperCase()}
                            </span>

                            <span class="news-status"
                                style="color:${statusColor};">
                                ${item.status}
                            </span>

                        </div>

                        <div class="news-title">
                            ${item.title}
                        </div>

                        <div class="news-meta">

                            <span>
                                Time: <strong>${item.time_uk}</strong>
                            </span>

                            <span>
                                Avoid (±5m):
                                <strong>${item.window_uk}</strong>
                            </span>

                        </div>

                    </div>

                `;

            }).join('');

        }


        // RENDER BOTH PAIR STATUSES

        function renderPairStatuses(pairData) {

            const container =
                document.getElementById('pair-status-container');

            if (!pairData || Object.keys(pairData).length === 0) {

                container.innerHTML = `
                    <div style="color:var(--muted);font-size:.8rem;">
                        Pair status unavailable.
                    </div>
                `;

                return;
            }

            container.innerHTML = Object.entries(pairData).map(
                ([pair, info]) => {

                    const healthy =
                        info.oanda_health &&
                        info.oanda_health.includes('Healthy');

                    const healthColor =
                        healthy ? '#00f5a0' : '#ffbd4a';

                    const dotClass =
                        healthy ? 'status-dot' : 'status-dot warning';

                    return `

                        <div class="pair-card">

                            <div class="pair-card-top">

                                <span class="pair-name">
                                    ${pair}
                                </span>

                                <span class="pair-health"
                                    style="color:${healthColor};">
                                    <span class="${dotClass}"
                                        style="width:7px;height:7px;margin-right:5px;">
                                    </span>
                                    ${info.oanda_health || 'Unknown'}
                                </span>

                            </div>

                            <div class="pair-info">
                                Scanner: ${info.scanner_status || 'Unknown'}
                            </div>

                            <div class="pair-info">
                                M5 Trend: ${info.m5_trend || 'Unknown'}
                                · ML: ${info.ml_status || 'Unknown'}
                            </div>

                            <div class="pair-last-alert">
                                Last alert: ${info.last_alert || 'None yet'}
                            </div>

                        </div>

                    `;

                }

            ).join('');

        }


        // UPDATE STATUS INDICATORS

        function updateStatusDot(dotId, status, isHealthy) {

            const dot = document.getElementById(dotId);

            if (!dot) return;

            dot.className = 'status-dot';

            if (!isHealthy) {
                dot.classList.add('warning');
            }

            if (
                status &&
                (
                    status.includes('Error') ||
                    status.includes('Disconnected') ||
                    status.includes('Failed')
                )
            ) {
                dot.className = 'status-dot offline';
            }

        }


        // FETCH DASHBOARD DATA

        function fetchData() {

            fetch('/api/data')

                .then(res => {
                    if (!res.ok) {
                        throw new Error("Dashboard API unavailable");
                    }
                    return res.json();
                })

                .then(data => {

                    if (!data.state) return;

                    // Health values

                    document.getElementById('oanda-health').innerText =
                        data.state.oanda_health || 'Unknown';

                    document.getElementById('tele-health').innerText =
                        data.state.telegram_health || 'Unknown';

                    document.getElementById('ml-health').innerText =
                        data.state.ml_status || 'Unknown';

                    document.getElementById('scan-task').innerText =
                        data.state.scanner_status || 'Unknown';

                    document.getElementById('m5-trend').innerText =
                        data.state.m5_trend || 'Unknown';


                    // Health colours

                    const oandaHealthy =
                        data.state.oanda_health &&
                        !data.state.oanda_health.includes('Error');

                    const telegramHealthy =
                        data.state.telegram_health &&
                        !data.state.telegram_health.includes('Error');

                    const mlHealthy =
                        data.state.ml_status &&
                        data.state.ml_status.includes('Loaded');

                    document.getElementById('oanda-health').style.color =
                        oandaHealthy ? '#00f5a0' : '#ff4d6d';

                    document.getElementById('tele-health').style.color =
                        telegramHealthy ? '#00f5a0' : '#ff4d6d';

                    document.getElementById('ml-health').style.color =
                        mlHealthy ? '#00f5a0' : '#ffbd4a';


                    // Animated health indicators

                    updateStatusDot(
                        'oanda-dot',
                        data.state.oanda_health,
                        oandaHealthy
                    );

                    updateStatusDot(
                        'tele-dot',
                        data.state.telegram_health,
                        telegramHealthy
                    );

                    updateStatusDot(
                        'ml-dot',
                        data.state.ml_status,
                        mlHealthy
                    );
                    renderPairStatuses(data.pairs);

                    // Calendar

                    renderNews(
                        data.state.todays_news,
                        data.state.symbol
                    );


                    // Logs - preserve user's scroll position

                    const logBox =
                        document.getElementById('log-container');

                    const isScrolledToBottom =
                        logBox.scrollHeight -
                        logBox.clientHeight <=
                        logBox.scrollTop + 5;

                    if (Array.isArray(data.logs)) {
                        logBox.innerHTML =
                            data.logs.join('<br>');
                    }

                    if (isScrolledToBottom) {
                        logBox.scrollTop = logBox.scrollHeight;
                    }

                })

                .catch(error => {
                    console.error("Dashboard refresh error:", error);
                });

        }


        // INITIAL LOAD AND AUTOMATIC REFRESH

        fetchData();

        setInterval(fetchData, 3000);

    </script>

</body>
</html>
"""


@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE, pairs=AVAILABLE_PAIRS, current_symbol=app_state["symbol"])

@app.route('/api/data')
def api_data():
    selected = app_state["symbol"]
    selected_status = pair_status[selected]
    app_state["oanda_health"] = selected_status["oanda_health"]
    app_state["scanner_status"] = selected_status["scanner_status"]
    app_state["m5_trend"] = selected_status["m5_trend"]
    app_state["ml_status"] = selected_status["ml_status"]
    app_state["todays_news"] = get_todays_pair_news(selected, buffer_minutes=5)
    return jsonify({"state": app_state, "pairs": pair_status, "logs": list(system_logs)})


@app.route('/api/settings', methods=['POST'])
def api_settings():
    data = request.get_json(silent=True) or {}
    new_symbol = data.get("symbol")
    if new_symbol in AVAILABLE_PAIRS:
        # Dashboard selector changes the displayed pair only, not scanner coverage.
        app_state["symbol"] = new_symbol
        log_msg(f"Dashboard view changed to {new_symbol}; both pairs remain active.")
        return jsonify({"status": "success", "symbol": new_symbol})
    return jsonify({"status": "error",
                    "message": "Invalid symbol. Only EUR_JPY and GBP_JPY are enabled."}), 400

if __name__ == '__main__':
    scanner_thread = threading.Thread(target=run_fvg_scanner, name="fvg-scanner", daemon=True)
    scanner_thread.start()
    app.run(host='0.0.0.0', port=8001, debug=False, use_reloader=False)