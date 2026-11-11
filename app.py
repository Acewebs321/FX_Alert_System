import requests
import pandas as pd
import numpy as np
import time
import os
import threading
import joblib
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from collections import deque
from flask import Flask, jsonify, request, render_template_string




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

# Locked exclusively to EUR_JPY as requested
AVAILABLE_PAIRS = ["EUR_JPY", "EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD", "USD_CHF", "GBP_JPY"]

OANDA_URL = "https://api-fxpractice.oanda.com"
TELEGRAM_TOKEN = "8641583463:AAGrsLR0Xmf-fOd3FXafjC8lbqUiqCasZGM"
TELEGRAM_CHAT_ID ="1110417286"
OANDA_TOKEN = "56c3289f164a1a64b30de44241b6ec5a-4aa1d8f08ad01339e4ee2f4d6be3d7c9"

SCAN_INTERVAL = 30
FVG_LOOKBACK = 10
M3_SMA_PERIOD = 20
M5_SMA_PERIOD = 50
FVG_LOOKAHEAD = 10
MODEL_PATH = "fvg_revisit_model.joblib"

news_cache = {"timestamp": None, "data": []}
pending_crossings = {}
alerted_setups = set()
ml_bundle = None

# Dashboard Log Storage
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
def load_ml_model():
    global ml_bundle
    if os.path.exists(MODEL_PATH):
        try:
            ml_bundle = joblib.load(MODEL_PATH)
            app_state["ml_status"] = "Loaded (EUR_JPY) 🟢"
            log_msg(f"ML model loaded successfully from {MODEL_PATH}.")
        except Exception as e:
            ml_bundle = None
            app_state["ml_status"] = "Load Error (Rule Fallback) 🟡"
            log_msg(f"Failed to load ML model ({e}); using statistical rule fallback.")
    else:
        ml_bundle = None
        app_state["ml_status"] = "Stats Fallback Mode 🟡"
        log_msg(f"{MODEL_PATH} not found; using EUR_JPY 1,817-row statistical lookup fallback.")

def evaluate_eurjpy_ml_setup(setup, fvg, completed_3m, alert_close_time):
    """
    Computes the exact 16 features used in training for EUR_JPY and returns:
    (revisit_prob, confidence_tier_label, is_recommended, feature_summary_dict)
    """
    crossing_time = setup["time"]
    fvg_time = fvg["time"]

    # Calculate SMAs on completed_3m (60 candles is >= 50 needed for SMA50)
    df_calc = completed_3m.copy()
    df_calc["SMA20"] = df_calc["Close"].rolling(M3_SMA_PERIOD).mean()
    df_calc["SMA50"] = df_calc["Close"].rolling(M5_SMA_PERIOD).mean()
    current = df_calc.iloc[-1]

    sma_3m_20 = float(current["SMA20"])
    sma_3m_50 = float(current["SMA50"]) if pd.notna(current["SMA50"]) else sma_3m_20

    # Use M5 SMA50 stored at crossing or fetch latest M5 SMA50
    sma_5m_50 = setup.get("m5_sma50")
    if sma_5m_50 is None or pd.isna(sma_5m_50):
        _, sma_5m_50 = get_m5_trend_and_sma()
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

    is_optimal_gap = int((gap_size > 0.006) and (gap_size <= 0.012))
    is_exhaustion_gap = int(gap_size > 0.024)

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

    # If the trained joblib model is present, use the exact XGBoost + Random Forest ensemble
    if ml_bundle is not None:
        try:
            feature_cols = ml_bundle["features"]
            X_live = pd.DataFrame([row_dict])[feature_cols]
            xgb_prob = float(ml_bundle["xgb"].predict_proba(X_live)[0, 1])
            rf_prob = float(ml_bundle["rf"].predict_proba(X_live)[0, 1])
            raw_score = 0.55 * xgb_prob + 0.45 * rf_prob

            p30 = ml_bundle.get("p30_cutoff", 0.45)
            p50 = ml_bundle.get("p50_cutoff", 0.50)
            p70 = ml_bundle.get("p70_cutoff", 0.55)

            if raw_score >= p70:
                tier = "🔥 HIGH CONVICTION (Top 30%)"
                est_wr = 0.848
            elif raw_score >= p50:
                tier = "✅ STRONG SETUP (Top 50%)"
                est_wr = 0.817
            elif raw_score >= p30:
                tier = "🟡 MODERATE SETUP (Top 70%)"
                est_wr = 0.801
            else:
                tier = "⚠️ LOW PROBABILITY (Bottom 30%)"
                est_wr = 0.705

            return est_wr, raw_score, tier, row_dict
        except Exception as e:
            log_msg(f"ML prediction error, using empirical lookup: {e}")

    # Empirical lookup fallback based directly on the 1,817-row EUR_JPY diagnostic table
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

def get_oanda_candles(granularity="M3", count=60):
    """Fetch candles and log useful details when OANDA rejects or cannot answer."""
    if not OANDA_TOKEN:
        app_state["oanda_health"] = "Missing Credentials"
        log_msg("OANDA token is missing. Set the OANDA_TOKEN variable.")
        return pd.DataFrame()

    # Locked to EUR_JPY
    url = f"{OANDA_URL}/v3/instruments/EUR_JPY/candles"
    headers = {
        "Authorization": f"Bearer {OANDA_TOKEN}",
        "Accept-Datetime-Format": "RFC3339",
    }
    params = {"granularity": granularity, "count": count, "price": "M"}

    try:
        response = requests.get(url, headers=headers, params=params, timeout=10)
        response.raise_for_status()

        payload = response.json()
        candles = []
        for candle in payload.get("candles", []):
            mid = candle.get("mid")
            if not mid:
                continue
            candles.append({
                "Time": pd.to_datetime(candle["time"], utc=True),
                "Open": float(mid["o"]),
                "High": float(mid["h"]),
                "Low": float(mid["l"]),
                "Close": float(mid["c"]),
                "Complete": candle.get("complete", False),
            })

        app_state["oanda_health"] = "Healthy 🟢"
        if not candles:
            log_msg(f"OANDA returned no usable candles (granularity={granularity}, symbol=EUR_JPY).")
            return pd.DataFrame()

        return pd.DataFrame(candles).set_index("Time").sort_index()

    except requests.HTTPError as e:
        response = e.response
        status_code = response.status_code if response is not None else "unknown"
        body = response.text[:1000] if response is not None else "No response body"
        app_state["oanda_health"] = f"HTTP Error {status_code} 🔴"
        log_msg(f"OANDA HTTP error | symbol=EUR_JPY | granularity={granularity} | status={status_code} | body={body}")
        return pd.DataFrame()

    except requests.Timeout as e:
        app_state["oanda_health"] = "Timeout 🔴"
        log_msg(f"OANDA timeout | symbol=EUR_JPY | granularity={granularity} | {e}")
        return pd.DataFrame()

    except requests.ConnectionError as e:
        app_state["oanda_health"] = "Connection Error 🔴"
        log_msg(f"OANDA connection error | symbol=EUR_JPY | granularity={granularity} | {e}")
        return pd.DataFrame()

    except requests.RequestException as e:
        app_state["oanda_health"] = "Request Error 🔴"
        log_msg(f"OANDA request error | symbol=EUR_JPY | granularity={granularity} | {e}")
        return pd.DataFrame()

    except (ValueError, KeyError, TypeError) as e:
        app_state["oanda_health"] = "Data Error 🔴"
        log_msg(f"OANDA response/data parsing error | granularity={granularity} | {type(e).__name__}: {e}")
        return pd.DataFrame()

def get_m5_trend_and_sma():
    df = get_oanda_candles("M5", count=100)
    if df.empty:
        return None, None
    df = df[df["Complete"]].copy()
    if len(df) < M5_SMA_PERIOD:
        return None, None
    df["SMA50"] = df["Close"].rolling(M5_SMA_PERIOD).mean()
    latest = df.iloc[-1]
    sma50_val = float(latest["SMA50"])
    if latest["Close"] > sma50_val:
        return "bullish", sma50_val
    if latest["Close"] < sma50_val:
        return "bearish", sma50_val
    return None, sma50_val

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

def find_matching_fvg(
    df,
    direction,
    crossing_time,
    lookback=10,
    lookahead=10
):
    df = df[df["Complete"]].copy().sort_index()

    if len(df) < 3 or crossing_time not in df.index:
        return None

    crossing_pos = df.index.get_loc(crossing_time)

    # Candidate FVGs are identified by the index of their
    # third candle, which confirms the FVG when it closes.
    first_i = max(2, crossing_pos - lookback)
    last_i = min(len(df) - 1, crossing_pos + lookahead)

    before_or_at = []
    after = []

    for i in range(first_i, last_i + 1):
        first = df.iloc[i - 2]
        second = df.iloc[i - 1]
        third = df.iloc[i]

        if direction == "bearish":
            correct_colours = (
                first["Close"] < first["Open"]
                and second["Close"] < second["Open"]
                and third["Close"] < third["Open"]
            )
            gap_exists = first["Low"] > third["High"]

            if correct_colours and gap_exists:
                candidate = {
                    "type": "Bearish FVG",
                    "time": df.index[i],
                    "top": float(first["Low"]),
                    "bottom": float(third["High"])
                }
            else:
                continue

        elif direction == "bullish":
            correct_colours = (
                first["Close"] > first["Open"]
                and second["Close"] > second["Open"]
                and third["Close"] > third["Open"]
            )
            gap_exists = first["High"] < third["Low"]

            if correct_colours and gap_exists:
                candidate = {
                    "type": "Bullish FVG",
                    "time": df.index[i],
                    "top": float(third["Low"]),
                    "bottom": float(first["High"])
                }
            else:
                continue
        else:
            raise ValueError("direction must be 'bullish' or 'bearish'")

        if i <= crossing_pos:
            before_or_at.append(candidate)
        else:
            after.append(candidate)

    if before_or_at:
        return max(before_or_at, key=lambda fvg: fvg["time"])

    if after:
        return min(after, key=lambda fvg: fvg["time"])

    return None

def is_trading_time():
    uk_time = datetime.now(ZoneInfo("Europe/London"))
    return 7 <= uk_time.hour < 20

def run_fvg_scanner():
    load_ml_model()
    last_processed_crossing = None

    while True:
        try:
            app_state["symbol"] = "EUR_JPY"

            if not is_trading_time():
                app_state["scanner_status"] = "Paused (Outside UK Trading Hours)"
                time.sleep(SCAN_INTERVAL)
                continue

            app_state["scanner_status"] = "Fetching M3 Candles for EUR_JPY..."
            df_3m = get_oanda_candles("M3", count=60)
            if df_3m.empty:
                app_state["scanner_status"] = "Waiting for OANDA Data..."
                time.sleep(SCAN_INTERVAL)
                continue

            completed_3m = df_3m[df_3m["Complete"]].copy().sort_index()
            if len(completed_3m) < M5_SMA_PERIOD:
                app_state["scanner_status"] = "Waiting for sufficient completed M3 candles..."
                time.sleep(SCAN_INTERVAL)
                continue

            candle_time = completed_3m.index[-1]

            app_state["scanner_status"] = "Checking SMA20 Crossings..."
            crossing = detect_sma20_cross(completed_3m)

            if crossing is not None and candle_time != last_processed_crossing:
                last_processed_crossing = candle_time

                app_state["scanner_status"] = "Checking M5 Trend for new crossing..."
                trend, m5_sma50_val = get_m5_trend_and_sma()
                app_state["m5_trend"] = trend.upper() if trend else "Neutral"

                if trend == "bullish" and crossing == "below":
                    alert_type, alert_label, fvg_direction = "long", "LONG PULLBACK", "bearish"
                elif trend == "bearish" and crossing == "above":
                    alert_type, alert_label, fvg_direction = "short", "SHORT PULLBACK", "bullish"
                else:
                    alert_type = None

                if alert_type:
                    pending_crossings[str(candle_time)] = {
                        "time": candle_time,
                        "crossing": crossing,
                        "trend": trend,
                        "m5_sma50": m5_sma50_val,
                        "alert_type": alert_type,
                        "alert_label": alert_label,
                        "fvg_direction": fvg_direction,
                        "price": float(completed_3m.iloc[-1]["Close"])
                    }
                    log_msg(f"New Crossing: {crossing.upper()} | Trend: {trend} | Need {fvg_direction} FVG.")

            # 3. Process Pending Crossings
            app_state["scanner_status"] = f"Processing {len(pending_crossings)} pending setups..."
            for key in list(pending_crossings.keys()):
                setup = pending_crossings[key]
                crossing_time = setup["time"]

                if crossing_time not in completed_3m.index:
                    pending_crossings.pop(key, None)
                    continue

                age = (len(completed_3m) - 1) - completed_3m.index.get_loc(crossing_time)
                if age > FVG_LOOKAHEAD:
                    pending_crossings.pop(key, None)
                    continue

                fvg = find_matching_fvg(
                    completed_3m,
                    direction=setup["fvg_direction"],
                    crossing_time=crossing_time,
                    lookback=FVG_LOOKBACK,
                    lookahead=FVG_LOOKAHEAD
                )

                if fvg is None:
                    continue

                # ==========================================
                # CHECK IF FVG IS ALREADY VISITED BEFORE ALERT
                # ==========================================
                post_fvg_candles = completed_3m.loc[completed_3m.index > fvg["time"]]
                is_mitigated = False

                if not post_fvg_candles.empty:
                    if fvg["type"] == "Bearish FVG":
                        if (post_fvg_candles["High"] >= fvg["bottom"]).any():
                            is_mitigated = True
                    elif fvg["type"] == "Bullish FVG":
                        if (post_fvg_candles["Low"] <= fvg["top"]).any():
                            is_mitigated = True

                if is_mitigated:
                    log_msg(f"Discarding setup: {fvg['type']} at {fvg['time'].strftime('%H:%M')} already visited.")
                    pending_crossings.pop(key, None)
                    continue
                # ==========================================

                setup_key = (setup["alert_type"], str(crossing_time), str(fvg["time"]))
                if setup_key in alerted_setups:
                    pending_crossings.pop(key, None)
                    continue

                # Evaluate ML Revisit Probability & Features at the close of the detection M3 candle
                alert_close_time = candle_time + timedelta(minutes=3)
                est_revisit_prob, ml_raw_score, confidence_tier, feat_info = evaluate_eurjpy_ml_setup(
                    setup, fvg, completed_3m, alert_close_time
                )

                wait_mins = feat_info["Wait_After_Cross_Mins"]
                gap_size = feat_info["FVG GAP"]
                regime_label = "Post-Cross FVG (81.1% Base)" if wait_mins > 3.0 else "Pre-Cross FVG (70.5% Base)"

                log_msg(
                    f"FVG confirmed at {fvg['time']} UTC | Crossing at {crossing_time} UTC | "
                    f"ML Prob: {est_revisit_prob:.1%} ({confidence_tier})"
                )

                fvg_uk = fvg["time"].tz_convert("Europe/London")
                crossing_uk = crossing_time.tz_convert("Europe/London")
                detected_uk = pd.Timestamp.now(tz="Europe/London")
                delay_seconds = (detected_uk - fvg_uk).total_seconds()

                message = (
                    f"🚨 EUR_JPY - {setup['alert_label']}\n\n"
                    f"• ML 60m Revisit Est: {est_revisit_prob:.1%}\n"
                    f"• Signal Tier: {confidence_tier}\n"
                    f"• Setup Regime: {regime_label}\n"
                    f"• Wait After Cross: {wait_mins:.0f} mins\n"
                    f"• Trend: {setup['trend'].upper()}\n"
                    f"• M3 crossing: {setup['crossing'].upper()} SMA20\n"
                    f"• FVG: {fvg['type']} ({fvg['bottom']:.3f} - {fvg['top']:.3f})\n"
                    f"• FVG Gap: {gap_size:.3f}\n"
                    f"• Time since close: {delay_seconds:.0f}s"
                )

                if send_telegram(message):
                    alerted_setups.add(setup_key)
                    pending_crossings.pop(key, None)
                    log_msg("Telegram alert sent successfully.")

            app_state["scanner_status"] = "Idle (Waiting for next scan interval)"

        except Exception as e:
            app_state["scanner_status"] = "Error in Scanner Loop"
            log_msg(f"Scanner error: {e}")

        time.sleep(SCAN_INTERVAL)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>EUR_JPY ML FVG Scanner</title>
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

        .pair-badge {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 12px 16px;
            background: var(--bg-base);
            border: 1px solid var(--accent);
            border-radius: 8px;
            margin-bottom: 24px;
            font-weight: 700;
        }

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
        <h1>EUR_JPY Trading Alert Dashboard (ML Enabled)</h1>
    </div>
    
    <div class="container">
        <!-- Control Panel -->
        <div class="panel panel-controls">
            <h2>Active Instrument</h2>
            <div class="pair-badge">
                <span>Locked Pair:</span>
                <span style="color: #60a5fa;">EUR_JPY (ML Calibrated)</span>
            </div>
            
            <h2>Live API & ML Health</h2>
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

            <h2 style="margin-top: 24px;">Today's EUR/JPY News (UK Time)</h2>
            <div id="news-container">Checking calendar...</div>
        </div>

        <!-- Logs Panel -->
        <div class="panel panel-logs">
            <h2>System Logs</h2>
            <div id="log-container">Loading logs...</div>
        </div>
    </div>

    <script>
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
    return render_template_string(HTML_TEMPLATE, pairs=AVAILABLE_PAIRS, current_symbol="EUR_JPY")

@app.route('/api/data')
def api_data():
    app_state["todays_news"] = get_todays_pair_news("EUR_JPY", buffer_minutes=5)
    return jsonify({
        "state": app_state,
        "logs": list(system_logs)
    })

@app.route('/api/settings', methods=['POST'])
def api_settings():
    # Locked exclusively to EUR_JPY so the EUR_JPY ML model is never applied to another pair
    app_state["symbol"] = "EUR_JPY"
    return jsonify({"status": "success", "symbol": "EUR_JPY"})

if __name__ == '__main__':
    scanner_thread = threading.Thread(target=run_fvg_scanner, daemon=True)
    scanner_thread.start()
    app.run(host='0.0.0.0', port=8001, debug=False, use_reloader=False)