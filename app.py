import requests
import pandas as pd
import time
import threading
import asyncio
from datetime import datetime
from zoneinfo import ZoneInfo
from collections import deque
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from sse_starlette.sse import EventSourceResponse

# ==============================
# CONFIGURATION & STATE
# ==============================
SYMBOL = "EUR_JPY"
OANDA_URL = "https://api-fxpractice.oanda.com"
TELEGRAM_TOKEN = "" # Fill in
TELEGRAM_CHAT_ID = "" # Fill in
OANDA_TOKEN = "" # Fill in

SCAN_INTERVAL = 30
FVG_LOOKBACK = 10
M3_SMA_PERIOD = 20
M5_SMA_PERIOD = 50
FVG_LOOKAHEAD = 10

pending_crossings = {}
alerted_setups = set()

# Dashboard Log Storage
MAX_LOGS = 100
system_logs = deque(maxlen=MAX_LOGS)
new_log_event = asyncio.Event()

app = FastAPI()

# ==============================
# LOGGING INTERCEPTOR
# ==============================
def log_msg(msg: str):
    """Replaces standard print to both output to console and UI."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    formatted_msg = f"[{timestamp}] {msg}"
    print(formatted_msg, flush=True)
    system_logs.append(formatted_msg)
    
    # Notify connected web clients in the main event loop
    try:
        loop = asyncio.get_running_loop()
        loop.call_soon_threadsafe(new_log_event.set)
    except RuntimeError:
        pass # Loop not running yet

# ==============================
# EXISTING SCANNER LOGIC (Unchanged constraints)
# ==============================
def send_telegram(message):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        log_msg("Telegram credentials are missing.")
        return False
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try:
        response = requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": message}, timeout=10)
        response.raise_for_status()
        if not response.json().get("ok"):
            log_msg(f"Telegram rejected message: {response.json()}")
            return False
        return True
    except requests.RequestException as e:
        log_msg(f"Telegram error: {e}")
        return False

def get_oanda_candles(granularity="M3", count=60):
    url = f"{OANDA_URL}/v3/instruments/{SYMBOL}/candles"
    headers = {"Authorization": f"Bearer {OANDA_TOKEN}", "Accept-Datetime-Format": "RFC3339"}
    params = {"granularity": granularity, "count": count, "price": "M"}
    try:
        response = requests.get(url, headers=headers, params=params, timeout=10)
        response.raise_for_status()
        candles = []
        for candle in response.json().get("candles", []):
            mid = candle.get("mid")
            if not mid: continue
            candles.append({
                "Time": pd.to_datetime(candle["time"], utc=True),
                "Open": float(mid["o"]), "High": float(mid["h"]),
                "Low": float(mid["l"]), "Close": float(mid["c"]),
                "Complete": candle.get("complete", False)
            })
        if not candles: return pd.DataFrame()
        df = pd.DataFrame(candles)
        return df.set_index("Time").sort_index()
    except Exception as e:
        log_msg(f"OANDA API error: {e}")
        return pd.DataFrame()

def get_m5_trend():
    df = get_oanda_candles("M5", count=100)
    if df.empty: return None
    df = df[df["Complete"]].copy()
    if len(df) < M5_SMA_PERIOD: return None
    df["SMA50"] = df["Close"].rolling(M5_SMA_PERIOD).mean()
    latest = df.iloc[-1]
    if latest["Close"] > latest["SMA50"]: return "bullish"
    if latest["Close"] < latest["SMA50"]: return "bearish"
    return None

def detect_sma20_cross(df):
    df = df[df["Complete"]].copy()
    if len(df) < M3_SMA_PERIOD + 1: return None
    df["SMA20"] = df["Close"].rolling(M3_SMA_PERIOD).mean()
    previous, current = df.iloc[-2], df.iloc[-1]
    if previous["Close"] <= previous["SMA20"] and current["Close"] > current["SMA20"]: return "above"
    if previous["Close"] >= previous["SMA20"] and current["Close"] < current["SMA20"]: return "below"
    return None

def find_matching_fvg(df, direction, crossing_time, lookback=10, lookahead=10):
    df = df[df["Complete"]].copy().sort_index()
    if len(df) < 3 or crossing_time not in df.index: return None
    crossing_pos = df.index.get_loc(crossing_time)
    first_i = max(2, crossing_pos - lookback)
    last_i = min(len(df) - 1, crossing_pos + lookahead)
    before_or_at, after = [], []

    for i in range(first_i, last_i + 1):
        first, second, third = df.iloc[i - 2], df.iloc[i - 1], df.iloc[i]
        if direction == "bearish":
            valid_candles = all([first["Close"] < first["Open"], second["Close"] < second["Open"], third["Close"] < third["Open"]])
            valid_gap = first["Low"] > third["High"]
            if valid_candles and valid_gap:
                fvg = {"type": "Bearish FVG", "time": df.index[i], "top": float(first["Low"]), "bottom": float(third["High"])}
                (before_or_at if i <= crossing_pos else after).append(fvg)
        elif direction == "bullish":
            valid_candles = all([first["Close"] > first["Open"], second["Close"] > second["Open"], third["Close"] > third["Open"]])
            valid_gap = first["High"] < third["Low"]
            if valid_candles and valid_gap:
                fvg = {"type": "Bullish FVG", "time": df.index[i], "top": float(third["Low"]), "bottom": float(first["High"])}
                (before_or_at if i <= crossing_pos else after).append(fvg)

    if before_or_at: return max(before_or_at, key=lambda fvg: fvg["time"])
    if after: return min(after, key=lambda fvg: fvg["time"])
    return None

def is_trading_time():
    uk_time = datetime.now(ZoneInfo("Europe/London"))
    return 7 <= uk_time.hour < 20

def run_fvg_scanner():
    last_processed_crossing = None
    log_msg("Scanner background loop started.")
    while True:
        try:
            if not is_trading_time():
                log_msg("Outside UK trading hours (07:00–20:00).")
                time.sleep(SCAN_INTERVAL)
                continue

            df_3m = get_oanda_candles("M3", count=60)
            if df_3m.empty:
                log_msg("No M3 data received.")
                time.sleep(SCAN_INTERVAL)
                continue

            completed_3m = df_3m[df_3m["Complete"]].copy().sort_index()
            if len(completed_3m) < M3_SMA_PERIOD + 1:
                log_msg("Waiting for completed M3 candles.")
                time.sleep(SCAN_INTERVAL)
                continue

            candle_time = completed_3m.index[-1]
            current_uk = pd.Timestamp.now(tz="Europe/London")
            log_msg(f"Scan: UK {current_uk:%H:%M:%S} | Latest M3: {candle_time:%H:%M:%S} UTC")

            crossing = detect_sma20_cross(completed_3m)
            if crossing is not None and candle_time != last_processed_crossing:
                last_processed_crossing = candle_time
                trend = get_m5_trend()
                if trend == "bullish" and crossing == "below":
                    alert_type, alert_label, fvg_direction = "long", "LONG PULLBACK", "bearish"
                elif trend == "bearish" and crossing == "above":
                    alert_type, alert_label, fvg_direction = "short", "SHORT PULLBACK", "bullish"
                else:
                    alert_type = None

                if alert_type:
                    pending_crossings[str(candle_time)] = {
                        "time": candle_time, "crossing": crossing, "trend": trend,
                        "alert_type": alert_type, "alert_label": alert_label,
                        "fvg_direction": fvg_direction, "price": float(completed_3m.iloc[-1]["Close"])
                    }
                    log_msg(f"Crossing confirmed: {crossing.upper()} | Trend: {trend} | Need {fvg_direction} FVG.")

            for key in list(pending_crossings.keys()):
                setup = pending_crossings[key]
                crossing_time = setup["time"]
                
                if crossing_time not in completed_3m.index:
                    log_msg(f"Discarding missing crossing: {crossing_time}")
                    pending_crossings.pop(key, None)
                    continue
                
                crossing_pos = completed_3m.index.get_loc(crossing_time)
                latest_pos = len(completed_3m) - 1
                age = latest_pos - crossing_pos

                if age > FVG_LOOKAHEAD:
                    log_msg(f"Crossing expired: {crossing_time}")
                    pending_crossings.pop(key, None)
                    continue

                fvg = find_matching_fvg(completed_3m, direction=setup["fvg_direction"], crossing_time=crossing_time, lookback=FVG_LOOKBACK, lookahead=FVG_LOOKAHEAD)
                if fvg is None:
                    log_msg(f"Waiting for {setup['fvg_direction']} FVG | Age: {age} M3")
                    continue

                setup_key = (setup["alert_type"], str(crossing_time), str(fvg["time"]))
                if setup_key in alerted_setups:
                    pending_crossings.pop(key, None)
                    continue

                fvg_uk = fvg["time"].tz_convert("Europe/London")
                crossing_uk = crossing_time.tz_convert("Europe/London")
                detected_uk = pd.Timestamp.now(tz="Europe/London")
                delay_seconds = (detected_uk - fvg_uk).total_seconds()

                message = (
                    f"🚨 EURJPY - {setup['alert_label']}\n\n"
                    f"• Trend: {setup['trend'].upper()}\n"
                    f"• M3 crossing: {setup['crossing'].upper()} SMA20\n"
                    f"• FVG: {fvg['type']}\n"
                    f"• FVG completed (UK): {fvg_uk:%Y-%m-%d %H:%M:%S}\n"
                    f"• Crossing time (UK): {crossing_uk:%Y-%m-%d %H:%M:%S}\n"
                    f"• Detected (UK): {detected_uk:%H:%M:%S}\n"
                    f"• Time since close: {delay_seconds:.0f}s"
                )

                log_msg(f"FVG found at {fvg['time']} UTC; detected at {detected_uk:%H:%M:%S} UK.")
                if send_telegram(message):
                    alerted_setups.add(setup_key)
                    pending_crossings.pop(key, None)
                    log_msg("Telegram alert sent successfully.")
                else:
                    log_msg("Telegram failed; will retry.")

        except Exception as e:
            log_msg(f"Scanner error: {e}")

        time.sleep(SCAN_INTERVAL)

# ==============================
# FASTAPI DASHBOARD ROUTES
# ==============================
@app.on_event("startup")
def start_background_scanner():
    if not OANDA_TOKEN:
        print("WARNING: OANDA_TOKEN missing.")
    thread = threading.Thread(target=run_fvg_scanner, daemon=True)
    thread.start()

@app.get("/", response_class=HTMLResponse)
async def get_dashboard():
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>EUR/JPY FVG Scanner</title>
        <style>
            body { font-family: monospace; background-color: #1e1e1e; color: #d4d4d4; padding: 20px; }
            #logs { width: 100%; height: 80vh; overflow-y: scroll; background: #000; padding: 15px; border: 1px solid #333; }
            .status { color: #4af626; margin-bottom: 15px; font-size: 1.2em; }
        </style>
    </head>
    <body>
        <div class="status">🟢 Scanner Running Online</div>
        <div id="logs"></div>
        <script>
            const logsDiv = document.getElementById('logs');
            const source = new EventSource('/stream-logs');
            
            source.onmessage = function(event) {
                const newLog = document.createElement('div');
                newLog.textContent = event.data;
                logsDiv.appendChild(newLog);
                logsDiv.scrollTop = logsDiv.scrollHeight; // auto-scroll
            };
        </script>
    </body>
    </html>
    """
    return html_content

@app.get("/stream-logs")
async def stream_logs(request: Request):
    """Streams new log entries to the dashboard."""
    async def log_generator():
        # Yield the historical cache first
        for log in list(system_logs):
            yield {"data": log}
            
        # Stream new logs as they arrive
        last_log_count = len(system_logs)
        while True:
            if await request.is_disconnected():
                break
            await new_log_event.wait()
            new_log_event.clear()
            
            current_count = len(system_logs)
            if current_count >= last_log_count:
                # Get only the newest logs since last yield
                new_items = list(system_logs)[last_log_count - current_count:]
                for item in new_items:
                    yield {"data": item}
            last_log_count = len(system_logs)
            
    return EventSourceResponse(log_generator())

if __name__ == "__main__":
    import uvicorn
    # Run locally on your computer at http://127.0.0.1:8000
    uvicorn.run(app, host="0.0.0.0", port=8000)