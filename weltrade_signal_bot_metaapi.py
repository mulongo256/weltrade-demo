import os
import asyncio
from datetime import datetime, timezone, timedelta
import pandas as pd
import numpy as np
from metaapi_cloud_sdk import MetaApi

# SIGNAL ONLY. This program never creates, modifies or closes trades.
# It reads real candles from a Weltrade MT5 account through MetaApi Cloud.
#
# GitHub Secrets required:
# METAAPI_TOKEN   = MetaApi API token
# METAAPI_ACCOUNT_ID = MetaApi account id for your Weltrade MT5 account
# TELEGRAM_BOT_TOKEN
# TELEGRAM_CHAT_ID
#
# Optional:
# POLL_SECONDS=5
#
# Use the INVESTOR/READ-ONLY password when adding the Weltrade account
# to MetaApi. The bot does not need trading permission.

SYMBOLS = {
    "MAX GainX 2000", "MAX PainX 2000", "PainX 1200", "PainX 600",
    "PainX 800", "PainX 999", "GainX 600", "GainX 800", "GainX 999",
    "GainX 1200", "GainX 400", "MAX GainX 1000", "MAX PainX 1000",
    "PainX 400",
}

TFS = ["M1", "M2", "M3", "M4", "M5", "M15", "M20", "M30", "M45", "H1"]
META_TFS = {"M1":"1m","M2":"2m","M3":"3m","M4":"4m","M5":"5m",
            "M15":"15m","M20":"20m","M30":"30m","H1":"1h"}

TOKEN = os.environ["METAAPI_TOKEN"]
ACCOUNT_ID = os.environ["METAAPI_ACCOUNT_ID"]
TG_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
TG_CHAT = os.environ["TELEGRAM_CHAT_ID"]
POLL = int(os.getenv("POLL_SECONDS", "5"))

api = MetaApi(TOKEN)
last_alert = {}
locked = {}

async def tg(text):
    import urllib.request, urllib.parse
    data = urllib.parse.urlencode({"chat_id": TG_CHAT, "text": text}).encode()
    req = urllib.request.Request(
        f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage",
        data=data, method="POST"
    )
    await asyncio.to_thread(urllib.request.urlopen, req, timeout=20)

def typical(c):
    return (c["high"] + c["low"] + c["close"]) / 3.0

def weighted_close(c):
    return (c["high"] + c["low"] + 2*c["close"]) / 4.0

def rsi(series, period=10):
    d = series.diff()
    up = d.clip(lower=0)
    down = -d.clip(upper=0)
    # Wilder RMA, matching the usual MT5 RSI calculation.
    au = up.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    ad = down.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    rs = au / ad.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    out = out.where(ad != 0, 100.0)
    out = out.where(~((au == 0) & (ad == 0)), 50.0)
    return out

def lwma(series, period):
    w = np.arange(1, period + 1, dtype=float)
    return series.rolling(period).apply(lambda x: np.dot(x, w)/w.sum(), raw=True)

def ichimoku_green(df):
    # User's Ichimoku settings: Tenkan=1, Kijun=1, Span B=1.
    # With period 1, the green Tenkan line is the candle midpoint.
    return (df["high"] + df["low"]) / 2.0

def make_m45(m1):
    x = m1.copy()
    x["bucket"] = x["time"].dt.floor("45min")
    out = x.groupby("bucket", sort=True).agg(
        open=("open","first"), high=("high","max"),
        low=("low","min"), close=("close","last"),
        tickVolume=("tickVolume","sum")
    ).reset_index().rename(columns={"bucket":"time"})
    return out

def add_indicators(df):
    df = df.copy()
    df["rsi"] = rsi(typical(df), 10)
    df["green1_price"] = lwma(weighted_close(df), 7)
    df["green2_price"] = ichimoku_green(df)
    # The MT5 visual setup compares these lines with RSI's 0-100 levels.
    # Because price and RSI have different units, normalize each green line
    # within the same candle's rolling price range before applying 0-100 zones.
    lo = df["low"].rolling(100).min()
    hi = df["high"].rolling(100).max()
    span = (hi - lo).replace(0, np.nan)
    df["green1"] = 100 * (df["green1_price"] - lo) / span
    df["green2"] = 100 * (df["green2_price"] - lo) / span
    return df

async def candles(connection, symbol, tf, limit=1200):
    if tf == "M45":
        # M45 is built from real M1 candles; no fake data is generated.
        raw = await connection.get_historical_candles(
            symbol=symbol, timeframe="1m", limit=min(limit, 1000)
        )
        return make_m45(pd.DataFrame(raw))
    raw = await connection.get_historical_candles(
        symbol=symbol, timeframe=META_TFS[tf], limit=min(limit, 1000)
    )
    return pd.DataFrame(raw)

def agrees(row, side):
    if not all(pd.notna(row.get(k)) for k in ("rsi","green1","green2")):
        return False
    if side == "BUY":
        return 0 <= row.rsi <= 10 and 0 <= row.green1 <= 10 and 0 <= row.green2 <= 10
    return 90 <= row.rsi <= 100 and 90 <= row.green1 <= 100 and 90 <= row.green2 <= 100

async def check_symbol(connection, symbol):
    if symbol not in SYMBOLS:
        return
    results = {}
    for tf in TFS:
        df = await candles(connection, symbol, tf)
        if len(df) < 120:
            return
        df = add_indicators(df)
        # Use the latest completed candle for CONFIRMED signals.
        row = df.iloc[-2]
        results[tf] = row

    for side in ("BUY", "SELL"):
        count = sum(agrees(results[tf], side) for tf in TFS)
        if count != 10:
            continue

        candle_time = str(results["M1"]["time"])
        key = f"{symbol}|{side}|{candle_time}"
        if last_alert.get(symbol) == key or locked.get(symbol) == side:
            continue

        if side == "BUY":
            msg = (f"🟢 BUY CONFIRMED\nSymbol: {symbol}\nAgreement: 10/10\n"
                   "TP1: RSI 50\nInvalidation: RSI 0\n"
                   "Timeframes: M1, M2, M3, M4, M5, M15, M20, M30, M45, H1")
        else:
            msg = (f"🔴 SELL CONFIRMED\nSymbol: {symbol}\nAgreement: 10/10\n"
                   "TP1: RSI 50\nInvalidation: RSI 100\n"
                   "Timeframes: M1, M2, M3, M4, M5, M15, M20, M30, M45, H1")

        await tg(msg)
        last_alert[symbol] = key
        locked[symbol] = side

async def main():
    account = await api.metatrader_account_api.get_account(ACCOUNT_ID)
    if account.state != "DEPLOYED":
        await account.deploy()
    await account.wait_connected()

    conn = account.get_streaming_connection()
    await conn.connect()
    await conn.wait_synchronized()

    # Read-only market-data connection. No trade functions are called.
    broker_symbols = await conn.get_symbols()
    wanted = [s for s in SYMBOLS if s in broker_symbols]
    missing = sorted(SYMBOLS - set(wanted))

    await tg(
        "✅ Weltrade market-data connection online.\n"
        f"Connected symbols: {len(wanted)}/{len(SYMBOLS)}\n"
        + (f"Missing from MT5: {', '.join(missing)}" if missing else "All requested symbols found.")
    )

    for symbol in wanted:
        await conn.subscribe_to_market_data(symbol)

    while True:
        for symbol in wanted:
            try:
                await check_symbol(conn, symbol)
            except Exception as e:
                print(f"{symbol}: {e}")
        await asyncio.sleep(POLL)

if __name__ == "__main__":
    asyncio.run(main())
