"""
GitHub Actions Trading Scanner - Improved Version
- Variable Take Profits (TP1, TP2, TP3)
- Smarter Stop Loss
- Leverage Suggestion
- Beautiful Persian Telegram Messages
"""

import asyncio
import ccxt.async_support as ccxt
import pandas as pd
import numpy as np
import time
import logging
import sys
import os
import json
import base64
from datetime import datetime, timezone
import math
import aiohttp

try:
    import joblib
except Exception:
    joblib = None

GITHUB_REPO = os.environ.get('GITHUB_REPOSITORY', '')
GITHUB_TOKEN = os.environ.get('GITHUB_TOKEN', '')
GITHUB_ACTOR = os.environ.get('GITHUB_ACTOR', 'scanner-bot')

COIN_LIST = [
    'DOGE/USDT','SHIB/USDT','PEPE/USDT','WIF/USDT','BONK/USDT','FLOKI/USDT','MEME/USDT',
    'KOMA/USDT','DOGS/USDT','NEIROETH/USDT','1000RATS/USDT','ORDI/USDT','PIPPIN/USDT',
    'BAN/USDT','1000SHIB/USDT','OM/USDT','CHILLGUY/USDT','PONKE/USDT','BOME/USDT',
    'MYRO/USDT','PEOPLE/USDT','PENGU/USDT','SPX/USDT','1000BONK/USDT','PNUT/USDT',
    'FARTCOIN/USDT','HIPPO/USDT','AIXBT/USDT','BRETT/USDT','VINE/USDT','MOODENG/USDT',
    'MUBARAK/USDT','MEW/USDT','POPCAT/USDT','1000FLOKI/USDT','1000CAT/USDT','ACT/USDT',
    'SLERF/USDT','DEGEN/USDT','1000PEPE/USDT'
]

EXCHANGE_ORDER = ['binance','bybit','kucoin','mexc','gateio','okx','bitget','huobi']
SR_TIMEFRAMES = ['1d', '4h', '1h', '15m', '5m']
TIMEFRAME_WEIGHTS = {'1d':5,'4h':3,'1h':2,'15m':1.5,'5m':1}

PIVOT_WINDOW = 5
MIN_TOUCHES = 2
SR_TOLERANCE_ATR = 0.5
PROXIMITY_THRESHOLD_ATR = 1.5
MODEL_PATH = 'trained_model.pkl'
CSV_LOG_PATH = 'confirmed_signals_log.csv'
PERSIST_PATH = 'prev_signals.json'

TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID')

if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
    print("ERROR: TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID are required!")
    sys.exit(1)

RATE_LIMIT_DELAY = 0.5
MAX_RETRIES = 3
RETRY_DELAY = 2.0

SYMBOL_MAP = {
    '1000PEPE/USDT': {'mexc':'PEPE1000/USDT','gateio':'PEPE1000/USDT','bitget':'PEPE1000/USDT'},
    '1000BONK/USDT': {'mexc':'BONK1000/USDT'},
    '1000FLOKI/USDT': {'mexc':'FLOKI1000/USDT'},
    '1000SHIB/USDT': {'mexc':'SHIB1000/USDT'},
    '1000CAT/USDT': {'mexc':'CAT1000/USDT'},
    'WIF/USDT': {'okx': 'WIF-USDT'},
    'NEIROETH/USDT': {'okx': 'NEIRO-USDT'}
}

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

exchanges = {}
sr_cache = {}
prev_signals = {}

# ---------------- GitHub helpers ----------------
async def read_file_from_github(filename):
    if not GITHUB_TOKEN or not GITHUB_REPO:
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                return f.read()
        except:
            return None
    try:
        url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{filename}"
        headers = {'Authorization': f'token {GITHUB_TOKEN}', 'Accept': 'application/vnd.github.v3+json'}
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    return base64.b64decode(data['content']).decode('utf-8')
                return None
    except Exception as e:
        logger.error(f"GitHub read error: {e}")
        return None

async def write_file_to_github(filename, content, commit_message):
    if not GITHUB_TOKEN or not GITHUB_REPO:
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        except:
            return False
    try:
        get_url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{filename}"
        headers = {'Authorization': f'token {GITHUB_TOKEN}', 'Accept': 'application/vnd.github.v3+json'}
        sha = None
        async with aiohttp.ClientSession() as session:
            async with session.get(get_url, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    sha = data['sha']
        encoded_content = base64.b64encode(content.encode('utf-8')).decode('utf-8')
        commit_data = {
            'message': commit_message,
            'content': encoded_content,
            'committer': {'name': GITHUB_ACTOR, 'email': f'{GITHUB_ACTOR}@github.com'}
        }
        if sha:
            commit_data['sha'] = sha
        put_url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{filename}"
        async with aiohttp.ClientSession() as session:
            async with session.put(put_url, headers=headers, json=commit_data) as response:
                return response.status in [200, 201]
    except Exception as e:
        logger.error(f"GitHub write error: {e}")
        return False

# ---------------- Telegram ----------------
async def send_telegram_message(message):
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {'chat_id': TELEGRAM_CHAT_ID, 'text': message, 'parse_mode': 'HTML'}
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, timeout=10) as response:
                return response.status == 200
    except Exception as e:
        logger.error(f"Telegram error: {e}")
        return False

async def send_error_notification(error_msg):
    message = f"🚨 <b>خطا در اسکنر</b>\n\n{error_msg}\n\n⏰ {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
    await send_telegram_message(message)

def format_price(price):
    if price < 0.0001:
        return f"{price:.10f}"
    elif price < 0.01:
        return f"{price:.8f}"
    elif price < 1:
        return f"{price:.6f}"
    else:
        return f"{price:.4f}"

async def format_and_send_signals(confirmed_list):
    if not confirmed_list:
        return

    for signal in confirmed_list:
        direction = signal['direction']
        emoji = "🟢" if direction == "LONG" else "🔴"
        dir_text = "لانگ (خرید)" if direction == "LONG" else "شورت (فروش)"

        entry = signal['price']
        sl = signal.get('sl', 0)
        tp1 = signal.get('tp1', 0)
        tp2 = signal.get('tp2', 0)
        tp3 = signal.get('tp3', 0)
        leverage = signal.get('leverage', '۵x - ۱۰x')

        # محاسبه درصدها
        if direction == "LONG":
            risk_pct = ((entry - sl) / entry) * 100
            rr1 = (tp1 - entry) / (entry - sl) if (entry - sl) > 0 else 0
            rr2 = (tp2 - entry) / (entry - sl) if (entry - sl) > 0 else 0
            rr3 = (tp3 - entry) / (entry - sl) if (entry - sl) > 0 else 0
        else:
            risk_pct = ((sl - entry) / entry) * 100
            rr1 = (entry - tp1) / (sl - entry) if (sl - entry) > 0 else 0
            rr2 = (entry - tp2) / (sl - entry) if (sl - entry) > 0 else 0
            rr3 = (entry - tp3) / (sl - entry) if (sl - entry) > 0 else 0

        message = f"""
{emoji} <b>سیگنال تأیید شده</b> {emoji}

📌 <b>{signal['symbol']}</b>
📍 جهت: <b>{dir_text}</b>
🏦 صرافی: {signal['exchange'].upper()}

━━━━━━━━━━━━━━━━━━
💰 <b>قیمت ورود:</b> <code>{format_price(entry)}</code>

🛑 <b>حد ضرر:</b> <code>{format_price(sl)}</code>
  ریسک: <b>{risk_pct:.2f}٪</b>

🎯 <b>تارگت‌ها:</b>
• هدف ۱: <code>{format_price(tp1)}</code>  (R:R ≈ {rr1:.1f})
• هدف ۲: <code>{format_price(tp2)}</code>  (R:R ≈ {rr2:.1f})
• هدف ۳: <code>{format_price(tp3)}</code>  (R:R ≈ {rr3:.1f})

━━━━━━━━━━━━━━━━━━
⚡ <b>اهرم پیشنهادی:</b> {leverage}
📊 اطمینان: <b>{signal['confidence']*100:.0f}٪</b>
📉 RSI: {signal['rsi']:.1f}
🔄 تعداد تأیید: {signal['count']}

⏰ {datetime.now(timezone.utc).strftime('%Y-%m-%d  %H:%M')} UTC
"""
        await send_telegram_message(message.strip())

# ---------------- Persistence & CSV ----------------
async def load_prev_signals():
    global prev_signals
    try:
        content = await read_file_from_github(PERSIST_PATH)
        prev_signals = json.loads(content) if content else {}
    except:
        prev_signals = {}

async def save_prev_signals():
    try:
        content = json.dumps(prev_signals, ensure_ascii=False, indent=2)
        await write_file_to_github(PERSIST_PATH, content, f"Update signals - {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')}")
    except Exception as e:
        logger.error(f"Save error: {e}")

async def append_csv_row(row):
    header = ['datetime_utc','symbol','direction','exchange','confidence','price','rsi','count','sl','tp1','tp2','tp3','leverage']
    try:
        existing = await read_file_from_github(CSV_LOG_PATH)
        new_row = ','.join(str(row.get(col, '')) for col in header)
        content = (existing.rstrip() + '\n' + new_row) if existing else (','.join(header) + '\n' + new_row)
        await write_file_to_github(CSV_LOG_PATH, content, f"Signal: {row.get('symbol')}")
    except Exception as e:
        logger.error(f"CSV error: {e}")

# ---------------- Exchange helpers ----------------
def get_symbol_for_exchange(symbol, exchange_id):
    if symbol in SYMBOL_MAP and exchange_id in SYMBOL_MAP[symbol]:
        return SYMBOL_MAP[symbol][exchange_id]
    return symbol

async def create_exchange_client(exchange_id):
    try:
        cls = getattr(ccxt, exchange_id)
        client = cls({'enableRateLimit': True, 'timeout': 30000, 'rateLimit': 1200})
        try:
            await client.load_markets()
        except:
            pass
        return client
    except Exception as e:
        logger.error(f"Init {exchange_id} failed: {e}")
        return None

async def init_exchanges():
    for eid in EXCHANGE_ORDER:
        client = await create_exchange_client(eid)
        if client:
            exchanges[eid] = client
        await asyncio.sleep(0.4)

async def close_exchanges():
    for c in exchanges.values():
        try:
            await c.close()
        except:
            pass

# ---------------- Indicators & S/R ----------------
async def fetch_ohlcv_with_retry(exchange, symbol, timeframe='15m', limit=300):
    for attempt in range(MAX_RETRIES):
        try:
            mapped = get_symbol_for_exchange(symbol, exchange.id)
            await asyncio.sleep(RATE_LIMIT_DELAY)
            ohlcv = await exchange.fetch_ohlcv(mapped, timeframe=timeframe, limit=limit)
            if not ohlcv or len(ohlcv) < 30:
                return None
            df = pd.DataFrame(ohlcv, columns=['timestamp','open','high','low','close','volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
            return df.sort_values('timestamp').reset_index(drop=True)
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(RETRY_DELAY * (2 ** attempt))
            else:
                return None
    return None

def calculate_indicators(df):
    if df is None or len(df) < 30:
        return df
    def rsi(series, period=14):
        delta = series.diff()
        gain = delta.where(delta > 0, 0).rolling(window=period).mean()
        loss = (-delta).where(delta < 0, 0).rolling(window=period).mean()
        rs = gain / (loss + 1e-10)
        return 100 - (100 / (1 + rs))
    def wavetrend(df_in, n1=10, n2=11):
        ap = (df_in['high'] + df_in['low'] + df_in['close']) / 3
        esa = ap.ewm(span=n1).mean()
        d = (ap - esa).abs().ewm(span=n1).mean()
        ci = (ap - esa) / (0.015 * d + 1e-10)
        return ci.ewm(span=n2).mean()
    def cci(df_in, period=20):
        tp = (df_in['high'] + df_in['low'] + df_in['close']) / 3
        sma = tp.rolling(window=period).mean()
        mad = tp.rolling(window=period).apply(lambda x: np.abs(x - x.mean()).mean())
        return (tp - sma) / (0.015 * mad + 1e-10)
    def adx(df_in, period=14):
        high, low, close = df_in['high'], df_in['low'], df_in['close']
        up = high.diff()
        down = -low.diff()
        plus_dm = up.where((up > down) & (up > 0), 0)
        minus_dm = down.where((down > up) & (down > 0), 0)
        tr = pd.concat([high-low, (high-close.shift(1)).abs(), (low-close.shift(1)).abs()], axis=1).max(axis=1)
        plus_di = (plus_dm.rolling(window=period).mean() / (tr.rolling(window=period).mean() + 1e-10)) * 100
        minus_di = (minus_dm.rolling(window=period).mean() / (tr.rolling(window=period).mean() + 1e-10)) * 100
        dx = (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10) * 100
        return dx.rolling(window=period).mean()
    try:
        df['rsi'] = rsi(df['close'])
        df['rsi_short'] = rsi(df['close'], 9)
        df['wavetrend'] = wavetrend(df)
        df['cci'] = cci(df)
        df['adx'] = adx(df)
    except:
        for col in ['rsi','rsi_short','wavetrend','cci','adx']:
            df[col] = 50
    return df

def normalize_features(df):
    for c in ['rsi','wavetrend','cci','adx','rsi_short']:
        if c in df.columns:
            std = df[c].std()
            df[c + '_norm'] = np.tanh(df[c] / std) if std > 1e-6 else 0
    return df

def detect_pivots(highs, lows, window=PIVOT_WINDOW):
    ph, pl = [], []
    for i in range(window, len(highs)-window):
        if all(highs[i] >= highs[j] for j in range(i-window, i+window+1) if j != i):
            ph.append((i, highs[i]))
        if all(lows[i] <= lows[j] for j in range(i-window, i+window+1) if j != i):
            pl.append((i, lows[i]))
    return ph, pl

def cluster_levels(levels, atr_value, tolerance_multiplier=SR_TOLERANCE_ATR):
    if not levels or atr_value <= 0:
        return []
    tol = atr_value * tolerance_multiplier
    levels = sorted(levels, key=lambda x: x[1])
    clusters, cur = [], [levels[0]]
    for lvl in levels[1:]:
        if abs(lvl[1] - cur[-1][1]) <= tol:
            cur.append(lvl)
        else:
            clusters.append((np.mean([x[1] for x in cur]), len(cur)))
            cur = [lvl]
    if cur:
        clusters.append((np.mean([x[1] for x in cur]), len(cur)))
    return clusters

def calculate_sr_levels(df, timeframe):
    if df is None or len(df) < 30:
        return [], []
    df = df.copy()
    df['tr'] = np.maximum(df['high'] - df['low'], np.maximum((df['high'] - df['close'].shift(1)).abs(), (df['low'] - df['close'].shift(1)).abs()))
    atr = df['tr'].rolling(14).mean().iloc[-1]
    if pd.isna(atr) or atr <= 0:
        atr = (df['high'] - df['low']).mean()
    ph, pl = detect_pivots(df['high'].values, df['low'].values)
    resist = cluster_levels(ph, atr)
    support = cluster_levels(pl, atr)
    weight = TIMEFRAME_WEIGHTS.get(timeframe, 1)
    valid_resist = [(p, touches * weight, timeframe) for p, touches in resist if touches >= MIN_TOUCHES]
    valid_support = [(p, touches * weight, timeframe) for p, touches in support if touches >= MIN_TOUCHES]
    return valid_resist, valid_support

async def get_multi_timeframe_sr(exchange, symbol):
    all_r, all_s = [], []
    for tf in SR_TIMEFRAMES:
        limit = {'1d':100,'4h':200,'1h':250,'15m':300,'5m':300}.get(tf, 300)
        df = await fetch_ohlcv_with_retry(exchange, symbol, timeframe=tf, limit=limit)
        if df is not None:
            r, s = calculate_sr_levels(df, tf)
            all_r.extend(r)
            all_s.extend(s)
    all_r.sort(key=lambda x: x[1], reverse=True)
    all_s.sort(key=lambda x: x[1], reverse=True)
    return all_r, all_s

def check_sr_proximity(current_price, resistance_levels, support_levels, atr_value, signal_type):
    if atr_value <= 0 or math.isnan(atr_value):
        return True
    threshold = atr_value * PROXIMITY_THRESHOLD_ATR
    if signal_type == 'LONG':
        nearby = [r for r in resistance_levels if r[0] > current_price]
        if nearby and (min(nearby, key=lambda x: x[0])[0] - current_price) < threshold:
            return False
    else:
        nearby = [s for s in support_levels if s[0] < current_price]
        if nearby and (current_price - max(nearby, key=lambda x: x[0])[0]) < threshold:
            return False
    return True

def calculate_sl_tp_and_leverage(price, direction, atr, support_levels, resistance_levels):
    """محاسبه هوشمند SL + ۳ تارگت + پیشنهاد اهرم"""
    if atr is None or atr <= 0 or math.isnan(atr):
        atr = price * 0.025

    atr_pct = (atr / price) * 100

    if direction == 'LONG':
        # حد ضرر: زیر نزدیک‌ترین ساپورت + بافر
        nearby_supports = [s[0] for s in support_levels if s[0] < price]
        if nearby_supports:
            nearest_support = max(nearby_supports)
            sl = nearest_support - (0.25 * atr)
        else:
            sl = price - (2.2 * atr)   # فاصله بیشتر برای جلوگیری از استاپ زودرس

        # حداقل فاصله حد ضرر
        min_sl_distance = price * 0.012   # حداقل ۱.۲٪
        if (price - sl) < min_sl_distance:
            sl = price - min_sl_distance

        risk = price - sl

        # تارگت‌ها
        nearby_resists = sorted([r[0] for r in resistance_levels if r[0] > price])
        tp1 = nearby_resists[0] if nearby_resists else price + (1.8 * risk)
        tp2 = nearby_resists[1] if len(nearby_resists) > 1 else price + (2.8 * risk)
        tp3 = nearby_resists[2] if len(nearby_resists) > 2 else price + (4.0 * risk)

        # اطمینان از R:R مناسب
        if (tp1 - price) < 1.5 * risk:
            tp1 = price + 1.6 * risk
        if (tp2 - price) < 2.4 * risk:
            tp2 = price + 2.6 * risk
        if (tp3 - price) < 3.5 * risk:
            tp3 = price + 3.8 * risk

    else:  # SHORT
        nearby_resists = [r[0] for r in resistance_levels if r[0] > price]
        if nearby_resists:
            nearest_resist = min(nearby_resists)
            sl = nearest_resist + (0.25 * atr)
        else:
            sl = price + (2.2 * atr)

        min_sl_distance = price * 0.012
        if (sl - price) < min_sl_distance:
            sl = price + min_sl_distance

        risk = sl - price

        nearby_supports = sorted([s[0] for s in support_levels if s[0] < price], reverse=True)
        tp1 = nearby_supports[0] if nearby_supports else price - (1.8 * risk)
        tp2 = nearby_supports[1] if len(nearby_supports) > 1 else price - (2.8 * risk)
        tp3 = nearby_supports[2] if len(nearby_supports) > 2 else price - (4.0 * risk)

        if (price - tp1) < 1.5 * risk:
            tp1 = price - 1.6 * risk
        if (price - tp2) < 2.4 * risk:
            tp2 = price - 2.6 * risk
        if (price - tp3) < 3.5 * risk:
            tp3 = price - 3.8 * risk

    # گرد کردن
    decimals = 8 if price < 1 else 6
    sl = round(sl, decimals)
    tp1 = round(tp1, decimals)
    tp2 = round(tp2, decimals)
    tp3 = round(tp3, decimals)

    # پیشنهاد اهرم بر اساس فاصله حد ضرر
    risk_pct = abs(price - sl) / price * 100

    if risk_pct < 1.3:
        leverage = "۱۵x – ۳۰x"
    elif risk_pct < 2.0:
        leverage = "۱۰x – ۲۰x"
    elif risk_pct < 3.0:
        leverage = "۷x – ۱۵x"
    elif risk_pct < 4.5:
        leverage = "۵x – ۱۰x"
    else:
        leverage = "۳x – ۷x"

    return sl, tp1, tp2, tp3, leverage

# ---------------- ML ----------------
def create_labels(df, forward_bars=4, threshold=0.002):
    returns = df['close'].shift(-forward_bars) / df['close'] - 1
    return np.where(returns >= threshold, 1, np.where(returns <= -threshold, -1, 0))

def lorentzian_knn_predict(df, k=8):
    feature_cols = ['rsi_norm','wavetrend_norm','cci_norm','adx_norm','rsi_short_norm']
    if df is None or len(df) < 60 or not all(c in df.columns for c in feature_cols):
        return 0, 0
    labels = create_labels(df)
    recent_df = df.iloc[-180:].copy()
    recent_labels = labels[-180:]
    current = recent_df[feature_cols].iloc[-1].values
    distances, valid_idx = [], []
    for i in range(0, len(recent_df)-6, 3):
        if pd.notna(recent_labels[i]) and recent_labels[i] != 0:
            hist = recent_df[feature_cols].iloc[i].values
            if not np.any(np.isnan(hist)) and not np.any(np.isnan(current)):
                d = np.sum(np.log(1 + np.abs(current - hist)))
                distances.append(d)
                valid_idx.append(i)
    if len(distances) < k:
        return 0, 0
    idxs = np.argsort(distances)[:k]
    neighbor_labels = [recent_labels[valid_idx[i]] for i in idxs]
    prediction_sum = sum(neighbor_labels)
    confidence = abs(np.mean(neighbor_labels))
    return prediction_sum, confidence

# ---------------- Main Scan ----------------
def _base_quote_from_unified(sym: str):
    try:
        base, right = sym.split('/')
        return base, right.split(':')[0]
    except:
        return sym, ''

async def _ensure_swap_options(exchange, eid: str):
    try:
        opts = getattr(exchange, 'options', {}) or {}
        if opts.get('defaultType') != 'swap':
            exchange.options = {**opts, 'defaultType': 'swap'}
            try:
                await exchange.load_markets()
            except:
                pass
    except:
        pass

def _resolve_swap_symbol(exchange, eid: str, requested_symbol: str):
    try:
        mapped = get_symbol_for_exchange(requested_symbol, eid)
        markets = getattr(exchange, 'markets', {}) or {}
        if mapped in markets and markets[mapped].get('swap'):
            return mapped
        if '/' in mapped:
            b, q = _base_quote_from_unified(mapped)
            colon = f"{b}/{q}:{q}"
            if colon in markets and markets[colon].get('swap'):
                return colon
        b, q = _base_quote_from_unified(mapped)
        for m in markets.values():
            if m.get('swap') and m.get('base') == b and m.get('quote') == q:
                return m.get('symbol')
    except:
        pass
    return None

def _pick_supported_timeframe(exchange):
    tfs = getattr(exchange, 'timeframes', {}) or {}
    for tf in ['15m', '5m', '1h', '4h']:
        if tf in tfs:
            return tf
    return '15m'

async def scan_market_single_run():
    logger.info("Starting market scan...")
    sr_cache.clear()
    for eid, ex in exchanges.items():
        await _ensure_swap_options(ex, eid)

    current_map = {}
    confirmed_list = []
    now_iso = datetime.now(timezone.utc).isoformat()

    for i, requested_symbol in enumerate(COIN_LIST):
        logger.info(f"Processing {requested_symbol} ({i+1}/{len(COIN_LIST)})")
        for eid in EXCHANGE_ORDER:
            exchange = exchanges.get(eid)
            if not exchange:
                continue
            resolved = _resolve_swap_symbol(exchange, eid, requested_symbol)
            if not resolved:
                continue
            tf = _pick_supported_timeframe(exchange)
            df = await fetch_ohlcv_with_retry(exchange, resolved, timeframe=tf, limit=300)
            if df is None:
                continue

            df = calculate_indicators(df)
            df = normalize_features(df)
            prediction_sum, confidence = lorentzian_knn_predict(df)

            if confidence < 0.6 or abs(prediction_sum) < 2:
                continue

            current_price = float(df['close'].iloc[-1])
            current_rsi = float(df['rsi'].iloc[-1]) if 'rsi' in df.columns else 50.0
            signal_type = 'LONG' if prediction_sum > 0 else 'SHORT'
            rsi_agrees = (current_rsi < 72) if signal_type == 'LONG' else (current_rsi > 28)
            if not rsi_agrees:
                continue

            cache_key = f"{requested_symbol}_{eid}"
            if cache_key not in sr_cache:
                resist, support = await get_multi_timeframe_sr(exchange, resolved)
                sr_cache[cache_key] = (resist, support)
            else:
                resist, support = sr_cache[cache_key]

            df['tr'] = np.maximum(df['high'] - df['low'], np.maximum((df['high'] - df['close'].shift(1)).abs(), (df['low'] - df['close'].shift(1)).abs()))
            atr = df['tr'].rolling(14).mean().iloc[-1]
            if pd.isna(atr):
                atr = (df['high'] - df['low']).mean()

            if not check_sr_proximity(current_price, resist, support, float(atr), signal_type):
                continue

            sl, tp1, tp2, tp3, leverage = calculate_sl_tp_and_leverage(current_price, signal_type, float(atr), support, resist)

            current_map[requested_symbol] = {
                'symbol': requested_symbol,
                'exchange': eid,
                'direction': signal_type,
                'rsi': current_rsi,
                'confidence': float(confidence),
                'price': current_price,
                'sl': sl,
                'tp1': tp1,
                'tp2': tp2,
                'tp3': tp3,
                'leverage': leverage,
                'ts': now_iso
            }
            logger.info(f"Signal found: {requested_symbol} {signal_type}")
            break
        await asyncio.sleep(0.08)

    # Confirmation logic
    for sym, cur in current_map.items():
        prev = prev_signals.get(sym)
        if prev and prev.get('direction') == cur['direction']:
            new_count = prev.get('count', 1) + 1
            prev_signals[sym] = {**cur, 'count': new_count, 'last_seen': cur['ts']}
        else:
            prev_signals[sym] = {**cur, 'count': 1, 'last_seen': cur['ts']}

    for sym, data in prev_signals.items():
        if sym in current_map and data.get('count', 0) >= 2:
            confirmed_list.append({
                'symbol': sym,
                'exchange': data['exchange'],
                'direction': data['direction'],
                'rsi': data.get('rsi', 50),
                'confidence': data['confidence'],
                'price': data['price'],
                'sl': data.get('sl'),
                'tp1': data.get('tp1'),
                'tp2': data.get('tp2'),
                'tp3': data.get('tp3'),
                'leverage': data.get('leverage', '۵x - ۱۰x'),
                'count': data['count']
            })
            await append_csv_row({
                'datetime_utc': now_iso,
                'symbol': sym,
                'direction': data['direction'],
                'exchange': data['exchange'],
                'confidence': data['confidence'],
                'price': data['price'],
                'rsi': data.get('rsi', ''),
                'count': data['count'],
                'sl': data.get('sl', ''),
                'tp1': data.get('tp1', ''),
                'tp2': data.get('tp2', ''),
                'tp3': data.get('tp3', ''),
                'leverage': data.get('leverage', '')
            })

    missing = [s for s in list(prev_signals.keys()) if s not in current_map]
    for s in missing:
        prev_signals.pop(s, None)

    await save_prev_signals()

    if confirmed_list:
        await format_and_send_signals(confirmed_list)
    else:
        logger.info("No confirmed signals")

    return confirmed_list

async def main():
    try:
        start_time = time.time()
        logger.info("="*50)
        logger.info("SCANNER STARTED (Improved Version)")
        await load_prev_signals()
        await init_exchanges()
        if not exchanges:
            raise Exception("No exchanges available")
        confirmed = await scan_market_single_run()
        await close_exchanges()
        logger.info(f"Completed in {time.time()-start_time:.1f}s | Signals: {len(confirmed)}")
    except Exception as e:
        logger.error(str(e))
        await send_error_notification(str(e))
        raise

if __name__ == '__main__':
    try:
        asyncio.run(main())
        sys.exit(0)
    except Exception as e:
        logger.error(f"Fatal: {e}")
        sys.exit(1)
