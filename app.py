import requests
import pandas as pd
import time
import threading
from datetime import datetime
from zoneinfo import ZoneInfo
from collections import deque
from flask import Flask, jsonify, request, render_template_string

# ==============================
# CONFIGURATION & GLOBAL STATE
# ==============================
# Application state dictionary to share data between the web server and background scanner
app_state = {
    "symbol": "EUR_JPY",
    "oanda_health": "Waiting...",
    "telegram_health": "Waiting...",
    "scanner_status": "Starting up...",
    "m5_trend": "Unknown"
}

# Supported pairs for the dropdown
AVAILABLE_PAIRS = ["EUR_JPY", "EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD", "USD_CHF", "GBP_JPY"]

OANDA_URL = "https://api-fxpractice.oanda.com"
TELEGRAM_TOKEN = "8641583463:AAGrsLR0Xmf-fOd3FXafjC8lbqUiqCasZGM" # Fill in
TELEGRAM_CHAT_ID = "1110417286" # Fill in
OANDA_TOKEN = "56c3289f164a1a64b30de44241b6ec5a-4aa1d8f08ad01339e4ee2f4d6be3d7c9" # Fill in

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
    url = f"{OANDA_URL}/v3/instruments/{app_state['symbol']}/candles"
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
        app_state["oanda_health"] = "Healthy 🟢"
        if not candles: return pd.DataFrame()
        df = pd.DataFrame(candles)
        return df.set_index("Time").sort_index()
    except Exception as e:
        app_state["oanda_health"] = "Error 🔴"
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
            # All three candles must be bearish.
            correct_colours = (
                first["Close"] < first["Open"]
                and second["Close"] < second["Open"]
                and third["Close"] < third["Open"]
            )

            # Bearish gap: candle 1 low > candle 3 high.
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
            # All three candles must be bullish.
            correct_colours = (
                first["Close"] > first["Open"]
                and second["Close"] > second["Open"]
                and third["Close"] > third["Open"]
            )

            # Bullish gap: candle 1 high < candle 3 low.
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
            raise ValueError(
                "direction must be 'bullish' or 'bearish'"
            )

        # Separate FVGs confirmed before/at and after crossing.
        if i <= crossing_pos:
            before_or_at.append(candidate)
        else:
            after.append(candidate)

    # An existing qualifying FVG can satisfy the setup immediately.
    if before_or_at:
        return max(before_or_at, key=lambda fvg: fvg["time"])

    # Otherwise wait for the earliest qualifying FVG after crossing.
    if after:
        return min(after, key=lambda fvg: fvg["time"])

    return None


def is_trading_time():
    uk_time = datetime.now(ZoneInfo("Europe/London"))
    return 7 <= uk_time.hour < 20

def run_fvg_scanner():
    last_processed_crossing = None
   

    
    while True:
        try:
            if not is_trading_time():
                app_state["scanner_status"] = "Paused (Outside UK Trading Hours)"
                time.sleep(SCAN_INTERVAL)
                continue

            app_state["scanner_status"] = f"Fetching M3 Candles for {app_state['symbol']}..."
            df_3m = get_oanda_candles("M3", count=60)
            if df_3m.empty:
                app_state["scanner_status"] = "Waiting for OANDA Data..."
                time.sleep(SCAN_INTERVAL)
                continue

            completed_3m = df_3m[df_3m["Complete"]].copy().sort_index()
            if len(completed_3m) < M3_SMA_PERIOD + 1:
                app_state["scanner_status"] = "Waiting for sufficient completed M3 candles..."
                time.sleep(SCAN_INTERVAL)
                continue

            candle_time = completed_3m.index[-1]
            
            # 1. Evaluate Trend
            app_state["scanner_status"] = "Checking M5 Trend..."
            trend = get_m5_trend()
            app_state["m5_trend"] = trend.upper() if trend else "Neutral"

            # 2. Evaluate Indicators
            app_state["scanner_status"] = "Checking SMA20 Crossings..."
            crossing = detect_sma20_cross(completed_3m)
            
            if crossing is not None and candle_time != last_processed_crossing:
                last_processed_crossing = candle_time
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

                # Only log after a qualifying FVG has been found.
                log_msg(
                    f"FVG confirmed at {fvg['time']} UTC | "
                    f"Crossing at {crossing_time} UTC | "
                    f"Detected at {pd.Timestamp.now(tz='UTC')} UTC"
                )


                setup_key = (setup["alert_type"], str(crossing_time), str(fvg["time"]))
                if setup_key in alerted_setups:
                    pending_crossings.pop(key, None)
                    continue

                fvg_uk = fvg["time"].tz_convert("Europe/London")
                crossing_uk = crossing_time.tz_convert("Europe/London")
                detected_uk = pd.Timestamp.now(tz="Europe/London")
                delay_seconds = (detected_uk - fvg_uk).total_seconds()

                message = (
                    f"🚨 {app_state['symbol']} - {setup['alert_label']}\n\n"
                    f"• Trend: {setup['trend'].upper()}\n"
                    f"• M3 crossing: {setup['crossing'].upper()} SMA20\n"
                    f"• FVG: {fvg['type']}\n"
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

# ==============================
# FLASK WEB DASHBOARD & API
# ==============================
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Trading Alert System</title>
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #121212; color: #e0e0e0; margin: 0; padding: 20px; }
        .container { display: flex; gap: 20px; flex-wrap: wrap; }
        .panel { background-color: #1e1e1e; border: 1px solid #333; padding: 20px; border-radius: 8px; flex: 1; min-width: 300px; }
        h2 { color: #ffffff; border-bottom: 1px solid #444; padding-bottom: 10px; margin-top: 0; }
        .status-row { display: flex; justify-content: space-between; margin-bottom: 10px; border-bottom: 1px dashed #333; padding-bottom: 5px; }
        .status-label { font-weight: bold; color: #aaa; }
        .status-val { color: #4af626; }
        #log-container { background-color: #000; border: 1px solid #333; padding: 10px; height: 400px; overflow-y: auto; font-family: monospace; white-space: pre-wrap; font-size: 0.9em; }
        select, button { padding: 8px; background: #333; color: white; border: 1px solid #555; border-radius: 4px; }
        button { cursor: pointer; background: #007bff; border: none; font-weight: bold; }
        button:hover { background: #0056b3; }
    </style>
</head>
<body>
    <h1 style="color: white; margin-bottom: 20px;">Scanner Dashboard</h1>
    
    <div class="container">
        <!-- Control Panel -->
        <div class="panel">
            <h2>System Control</h2>
            <div style="margin-bottom: 20px;">
                <label class="status-label">Active Trading Pair:</label>
                <select id="symbol-select">
                    {% for pair in pairs %}
                    <option value="{{ pair }}" {% if pair == current_symbol %}selected{% endif %}>{{ pair }}</option>
                    {% endfor %}
                </select>
                <button onclick="updateSymbol()">Apply Change</button>
            </div>
            
            <h2>Live API Health</h2>
            <div class="status-row"><span class="status-label">OANDA Status:</span><span class="status-val" id="oanda-health">...</span></div>
            <div class="status-row"><span class="status-label">Telegram Status:</span><span class="status-val" id="tele-health">...</span></div>
            
            <h2>Scanner Activity</h2>
            <div class="status-row"><span class="status-label">Current Task:</span><span class="status-val" id="scan-task" style="color: #ffaa00;">...</span></div>
            <div class="status-row"><span class="status-label">M5 Trend Filter:</span><span class="status-val" id="m5-trend">...</span></div>
        </div>

        <!-- Logs Panel -->
        <div class="panel" style="flex: 2;">
            <h2>System Logs</h2>
            <div id="log-container">Loading logs...</div>
        </div>
    </div>

    <script>
        function updateSymbol() {
            const newSymbol = document.getElementById("symbol-select").value;
            fetch('/api/settings', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ symbol: newSymbol })
            })
            .then(res => res.json())
            .then(data => alert("Successfully switched pair to: " + data.symbol));
        }

        function fetchData() {
            fetch('/api/data')
                .then(res => res.json())
                .then(data => {
                    // Update Status Indicators
                    document.getElementById('oanda-health').innerText = data.state.oanda_health;
                    document.getElementById('tele-health').innerText = data.state.telegram_health;
                    document.getElementById('scan-task').innerText = data.state.scanner_status;
                    document.getElementById('m5-trend').innerText = data.state.m5_trend;
                    
                    // Update Logs
                    const logBox = document.getElementById('log-container');
                    const isScrolledToBottom = logBox.scrollHeight - logBox.clientHeight <= logBox.scrollTop + 1;
                    logBox.innerHTML = data.logs.join('<br>');
                    if (isScrolledToBottom) {
                        logBox.scrollTop = logBox.scrollHeight;
                    }
                });
        }
        
        fetchData();
        setInterval(fetchData, 3000); // Refresh data every 3 seconds
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE, pairs=AVAILABLE_PAIRS, current_symbol=app_state["symbol"])

@app.route('/api/data')
def api_data():
    return jsonify({
        "state": app_state,
        "logs": list(system_logs)
    })

@app.route('/api/settings', methods=['POST'])
def api_settings():
    data = request.json
    new_symbol = data.get("symbol")
    if new_symbol and new_symbol in AVAILABLE_PAIRS:
        app_state["symbol"] = new_symbol
        # Clear out existing crossing memory to prevent crossing alerts across different pairs
        pending_crossings.clear()
        alerted_setups.clear()
        log_msg(f"=== SYSTEM SWITCHED PAIR TO {new_symbol} ===")
        return jsonify({"status": "success", "symbol": new_symbol})
    return jsonify({"status": "error", "message": "Invalid symbol"}), 400

if __name__ == '__main__':
    scanner_thread = threading.Thread(target=run_fvg_scanner, daemon=True)
    scanner_thread.start()
    app.run(host='127.0.0.1', port=8000, debug=False, use_reloader=False)