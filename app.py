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
SCAN_INTERVAL = 30
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
    Computes the 16 features for the active pair (EUR_JPY or GBP_JPY) using
    pair-specific gap quantiles and returns:
    (est_revisit_prob, raw_score, confidence_tier_label, feature_summary_dict)
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
    current = df_calc.iloc[-1]

    sma_3m_20 = float(current["SMA20"])
    sma_3m_50 = float(current["SMA50"]) if pd.notna(current["SMA50"]) else sma_3m_20

    sma_5m_50 = setup.get("m5_sma50")
    if sma_5m_50 is None or pd.isna(sma_5m_50):
        _, sma_5m_50 = get_m5_trend_and_sma(symbol)
    if sma_5m_50 is None or pd.isna(sma_5m_50):
        sma_5m_50 = sma_3m_50

    trend_bullish = 1 if setup["trend"] == "bullish" else 0
    sign = 1.0 if trend_bullish == 1 else -1.0

    fvg_to_cross_mins = (crossing_time - fvg_time).total_seconds() / 60.0
    wait_after_cross_mins = (alert_close_time - crossing_time).total_seconds() / 60.0
    gap_size = abs(float(fvg["top"]) - float(fvg["bottom"]))

    is_precross_fvg = int(wait_after_cross_mins <= 3.0)
    is_sweetspot_wait = int(round(wait_after_cross_mins, 1) in [6.0, 9.0, 15.0, 18.0, 21.0, 24.0])
    is_peak_wait_9_or_18 = int(round(wait_after_cross_mins, 1) in [9.0, 18.0])
    is_late_wait_27_30 = int(round(wait_after_cross_mins, 1) in [27.0, 30.0])

    is_optimal_gap = int((gap_size > q25_gap) and (gap_size <= q50_gap))
    is_exhaustion_gap = int(gap_size > q75_gap)

    hour = alert_close_time.hour
    is_ny_overlap = int(13 <= hour <= 16)
    is_late_session = int(hour >= 17)

    sma_3m_20_50_diff = (sma_3m_20 - sma_3m_50) * sign
    sma_3m20_5m50_diff = (sma_3m_20 - float(sma_5m_50)) * sign
    sma_3m50_5m50_diff = (sma_3m_50 - float(sma_5m_50)) * sign
    gap_to_3m_spread_ratio = gap_size / (abs(sma_3m_20_50_diff) + 1e-4)

    row_dict = {
        "Wait_After_Cross_Mins": wait_after_cross_mins,
        "FVG_to_Cross_Mins": fvg_to_cross_mins,
        "Is_PreCross_FVG": is_precross_fvg,
        "Is_SweetSpot_Wait": is_sweetspot_wait,
        "Is_Peak_Wait_9_or_18": is_peak_wait_9_or_18,
        "Is_Late_Wait_27_30": is_late_wait_27_30,
        "FVG GAP": gap_size,
        "Is_Optimal_Gap": is_optimal_gap,
        "Is_Exhaustion_Gap": is_exhaustion_gap,
        "Is_NY_Overlap": is_ny_overlap,
        "Is_Late_Session": is_late_session,
        "SMA_3m_20_50_Diff": sma_3m_20_50_diff,
        "SMA_3m20_5m50_Diff": sma_3m20_5m50_diff,
        "SMA_3m50_5m50_Diff": sma_3m50_5m50_diff,
        "Gap_to_3mSpread_Ratio": gap_to_3m_spread_ratio,
        "Trend_Bullish": trend_bullish
    }

    if bundle is not None:
        try:
            feature_cols = bundle["features"]
            X_live = pd.DataFrame([row_dict])[feature_cols]
            xgb_prob = float(bundle["xgb"].predict_proba(X_live)[0, 1])
            rf_prob = float(bundle["rf"].predict_proba(X_live)[0, 1])
            raw_score = 0.55 * xgb_prob + 0.45 * rf_prob

            p30 = bundle.get("p30_cutoff", 0.45)
            p50 = bundle.get("p50_cutoff", 0.50)
            p70 = bundle.get("p70_cutoff", 0.55)
            tier_wrs = bundle.get("tier_winrates", {})

            if raw_score >= p70:
                tier = "🔥 HIGH CONVICTION (Top 30%)"
                est_wr = tier_wrs.get("top30", 0.848)
            elif raw_score >= p50:
                tier = "✅ STRONG SETUP (Top 50%)"
                est_wr = tier_wrs.get("top50", 0.817)
            elif raw_score >= p30:
                tier = "🟡 MODERATE SETUP (Top 70%)"
                est_wr = tier_wrs.get("top70", 0.801)
            else:
                tier = "⚠️ LOW PROBABILITY (Bottom 30%)"
                est_wr = tier_wrs.get("bottom30", 0.705)

            return float(est_wr), raw_score, tier, row_dict
        except Exception as e:
            log_msg(f"ML prediction error for {symbol}, using empirical lookup: {e}")

    # Empirical fallback calibrated to JPY cross timing regimes
    wait_Empirical_Map = {
        3.0: 0.705, 6.0: 0.805, 9.0: 0.861, 12.0: 0.750,
        15.0: 0.832, 18.0: 0.860, 21.0: 0.804, 24.0: 0.826,
        27.0: 0.756, 30.0: 0.726, 33.0: 0.800
    }
    base_prob = wait_Empirical_Map.get(round(wait_after_cross_mins, 1), 0.783)
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
                f"🚨 {symbol} - {setup['alert_label']}\\n\\n"
                f"• ML 60m Revisit Est: {probability:.1%}\\n"
                f"• Signal Tier: {tier}\\n"
                f"• Setup Regime: {regime}\\n"
                f"• Wait After Cross: {wait_mins:.0f} mins\\n"
                f"• Trend (M5 SMA50): {setup['trend'].upper()}\\n"
                f"• M3 crossing: {setup['crossing'].upper()} SMA20\\n"
                f"• FVG: {fvg['type']} ({fvg['bottom']:.3f} - {fvg['top']:.3f})\\n"
                f"• FVG Gap: {gap_size:.3f}\\n"
                f"• Time since FVG candle: {delay_seconds:.0f}s"
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
    <title>ML Scanner</title>
    <style>
        :root {
            --bg-base: #0f172a;
            --bg-panel: #1e293b;
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
            --accent: #3b82f6;
            --accent-hover: #2563eb;
            --border: #334155;
            --log-bg: #020617;
            --log-text: #a3e635;
        }

        * { box-sizing: border-box; }

        body { 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif; 
            background-color: var(--bg-base); 
            color: var(--text-main); 
            margin: 0; 
            padding: 16px; 
            line-height: 1.5;
        }

        .header {
            margin-bottom: 24px;
            text-align: center;
        }

        .header h1 {
            margin: 0;
            font-size: 1.5rem;
            font-weight: 700;
        }

        .container { 
            display: flex; 
            flex-direction: column; 
            gap: 20px; 
            max-width: 1400px;
            margin: 0 auto;
        }

        @media (min-width: 992px) {
            .container { flex-direction: row; }
            .header { text-align: left; max-width: 1400px; margin: 0 auto 24px auto; }
            body { padding: 32px; }
        }

        .panel { 
            background-color: var(--bg-panel); 
            border: 1px solid var(--border); 
            padding: 24px; 
            border-radius: 12px; 
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        }

        .panel-controls { flex: 1; min-width: 320px; }
        .panel-logs { flex: 2; display: flex; flex-direction: column; }

        h2 { 
            color: var(--text-main); 
            font-size: 1.1rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            border-bottom: 1px solid var(--border); 
            padding-bottom: 12px; 
            margin-top: 0; 
            margin-bottom: 16px;
        }

        .control-group {
            display: flex;
            gap: 10px;
            margin-bottom: 24px;
            flex-wrap: wrap;
        }

        select { 
            flex: 1;
            padding: 10px; 
            background: var(--bg-base); 
            color: var(--text-main); 
            border: 1px solid var(--border); 
            border-radius: 6px; 
            font-size: 1rem;
            outline: none;
        }

        select:focus { border-color: var(--accent); }

        button { 
            padding: 10px 16px; 
            background: var(--accent); 
            color: white; 
            border: none; 
            border-radius: 6px;
            font-size: 0.95rem; 
            font-weight: 600; 
            cursor: pointer; 
            transition: background 0.2s; 
            flex-shrink: 0;
        }

        button:hover { background: var(--accent-hover); }

        .status-row { 
            display: flex; 
            justify-content: space-between; 
            align-items: center;
            padding: 12px 0; 
            border-bottom: 1px solid var(--border); 
        }

        .status-row:last-child { border-bottom: none; padding-bottom: 0; }

        .status-label { font-weight: 500; color: var(--text-muted); }
        .status-val { font-weight: 600; text-align: right; }

        #log-container { 
            background-color: var(--log-bg); 
            border: 1px solid var(--border); 
            border-radius: 8px;
            padding: 16px; 
            height: 60vh; 
            min-height: 300px;
            overflow-y: auto; 
            font-family: 'Consolas', 'Monaco', monospace; 
            white-space: pre-wrap; 
            font-size: 0.85rem; 
            color: var(--log-text);
            flex-grow: 1;
        }

        ::-webkit-scrollbar { width: 8px; }
        ::-webkit-scrollbar-track { background: var(--bg-base); border-radius: 4px; }
        ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 4px; }
        ::-webkit-scrollbar-thumb:hover { background: var(--text-muted); }
    </style>
</head>
<body>
    <div class="header">
        <h1>Trading Alert Dashboard</h1>
        <p style="color:#94a3b8;margin:6px 0 0;">EUR/JPY and GBP/JPY are scanned independently.</p>
    </div>
    
    <div class="container">
        <div class="panel panel-controls">
            <h2>Dashboard View (Both Pairs Scan)</h2>
            <div class="control-group">
                <select id="symbol-select">
                    {% for pair in pairs %}
                    <option value="{{ pair }}" {% if pair == current_symbol %}selected{% endif %}>{{ pair }}</option>
                    {% endfor %}
                </select>
                <button onclick="updateSymbol()">Apply</button>
            </div>
            
            <h2>Both Pair Scanners</h2>
            <div id="pair-status-container" style="margin-bottom: 24px; color: #94a3b8;">Loading pair status...</div>

            <h2>Selected Pair API & ML Health</h2>
            <div class="status-row">
                <span class="status-label">OANDA API:</span>
                <span class="status-val" id="oanda-health">...</span>
            </div>
            <div class="status-row">
                <span class="status-label">Telegram API:</span>
                <span class="status-val" id="tele-health">...</span>
            </div>
            <div class="status-row">
                <span class="status-label">ML Revisit Engine:</span>
                <span class="status-val" id="ml-health">...</span>
            </div>
            
            <h2 style="margin-top: 24px;">Scanner Activity</h2>
            <div class="status-row">
                <span class="status-label">Current Task:</span>
                <span class="status-val" id="scan-task" style="color: #fbbf24;">...</span>
            </div>
            <div class="status-row">
                <span class="status-label">M5 Trend Filter:</span>
                <span class="status-val" id="m5-trend">...</span>
            </div>

            <h2 style="margin-top: 24px;">Today's Pair News (UK Time)</h2>
            <div id="news-container">Checking calendar...</div>
        </div>

        <div class="panel panel-logs">
            <h2>System Logs</h2>
            <div id="log-container">Loading logs...</div>
        </div>
    </div>

    <script>
        function updateSymbol() {
            const newSymbol = document.getElementById("symbol-select").value;
            const btn = document.querySelector('button');
            const originalText = btn.innerText;
            btn.innerText = "Applying...";
            
            fetch('/api/settings', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ symbol: newSymbol })
            })
            .then(res => res.json())
            .then(data => {
                btn.innerText = originalText;
                const taskRow = document.getElementById('scan-task');
                taskRow.style.color = '#34d399';
                setTimeout(() => taskRow.style.color = '#fbbf24', 2000);
                fetchData();
            })
            .catch(() => {
                btn.innerText = "Error";
                setTimeout(() => btn.innerText = originalText, 2000);
            });
        }

        function renderNews(newsList, symbol) {
            const container = document.getElementById('news-container');
            if (!newsList || newsList.length === 0) {
                container.innerHTML = `<div style="color: #34d399; font-size: 0.9rem;">🟢 No High/Medium news today for ${symbol}</div>`;
                return;
            }

            container.innerHTML = newsList.map(item => {
                const badgeColor = item.impact === 'High' ? '#ef4444' : '#f59e0b';
                const statusColor = item.status === 'ACTIVE NOW' ? '#ef4444' : (item.status === 'Passed' ? '#64748b' : '#fbbf24');
                return `
                    <div style="background: #0f172a; border: 1px solid #334155; border-radius: 8px; padding: 10px; margin-bottom: 8px; font-size: 0.85rem;">
                        <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                            <span style="color: ${badgeColor}; font-weight: 700;">${item.currency} • ${item.impact.toUpperCase()}</span>
                            <span style="font-weight: 700; color: ${statusColor};">${item.status}</span>
                        </div>
                        <div style="font-weight: 600; margin-bottom: 4px;">${item.title}</div>
                        <div style="color: #94a3b8; display: flex; justify-content: space-between;">
                            <span>Time: <strong>${item.time_uk}</strong></span>
                            <span>Avoid (±5m): <strong>${item.window_uk}</strong></span>
                        </div>
                    </div>
                `;
            }).join('');
        }

        function renderPairStatuses(pairData) {
            const container = document.getElementById('pair-status-container');
            container.innerHTML = Object.entries(pairData || {}).map(([pair, info]) => `
                <div style="background:#0f172a;border:1px solid #334155;border-radius:8px;padding:10px;margin-bottom:8px;">
                    <div style="display:flex;justify-content:space-between;gap:8px;">
                        <strong style="color:#f8fafc;">${pair}</strong>
                        <span style="color:${info.oanda_health.includes('Healthy') ? '#34d399' : '#fbbf24'};">${info.oanda_health}</span>
                    </div>
                    <div style="font-size:.85rem;margin-top:5px;">${info.scanner_status}</div>
                    <div style="font-size:.8rem;margin-top:3px;">M5 trend: ${info.m5_trend} · ML: ${info.ml_status}</div>
                    <div style="font-size:.8rem;margin-top:3px;">Last alert: ${info.last_alert || 'None yet'}</div>
                </div>
            `).join('');
        }

        function fetchData() {
            fetch('/api/data')
                .then(res => res.json())
                .then(data => {
                    document.getElementById('oanda-health').innerText = data.state.oanda_health;
                    document.getElementById('tele-health').innerText = data.state.telegram_health;
                    document.getElementById('ml-health').innerText = data.state.ml_status;
                    document.getElementById('scan-task').innerText = data.state.scanner_status;
                    document.getElementById('m5-trend').innerText = data.state.m5_trend;
                    renderNews(data.state.todays_news, data.state.symbol);
                    
                    document.getElementById('oanda-health').style.color = data.state.oanda_health.includes('Error') ? '#ef4444' : '#34d399';
                    document.getElementById('tele-health').style.color = data.state.telegram_health.includes('Error') ? '#ef4444' : '#34d399';
                    document.getElementById('ml-health').style.color = data.state.ml_status.includes('Loaded') ? '#34d399' : '#fbbf24';
                    
                    const logBox = document.getElementById('log-container');
                    const isScrolledToBottom = logBox.scrollHeight - logBox.clientHeight <= logBox.scrollTop + 5;
                    logBox.innerHTML = data.logs.join('<br>');
                    
                    if (isScrolledToBottom) {
                        logBox.scrollTop = logBox.scrollHeight;
                    }
                });
        }
        
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