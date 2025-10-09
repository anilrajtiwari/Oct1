"""
GitHub Actions Trading Scanner - 15min automated signal generator

Features:
- Single execution for GitHub Actions cron (every 15 minutes)
- GitHub repository persistence (prev_signals.json, CSV logs)
- Environment variable support for tokens
- Enhanced error handling with Telegram alerts
- 40-coin support with exchange mapping
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

# Optional: joblib for external model
try:
    import joblib
except Exception:
    joblib = None

# GitHub repository management
GITHUB_REPO = os.environ.get('GITHUB_REPOSITORY', '')  # Automatically set by GitHub Actions
GITHUB_TOKEN = os.environ.get('GITHUB_TOKEN', '')    # GitHub token for commits
GITHUB_ACTOR = os.environ.get('GITHUB_ACTOR', 'scanner-bot')

# ---------------- CONFIG ----------------
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

# Remove 8h timeframe as requested
SR_TIMEFRAMES = ['1d', '4h', '1h', '15m', '5m']
TIMEFRAME_WEIGHTS = {'1d':5,'4h':3,'1h':2,'15m':1.5,'5m':1}

PIVOT_WINDOW = 5
MIN_TOUCHES = 2
SR_TOLERANCE_ATR = 0.5
PROXIMITY_THRESHOLD_ATR = 1.5
MODEL_PATH = 'trained_model.pkl'
CSV_LOG_PATH = 'confirmed_signals_log.csv'
PERSIST_PATH = 'prev_signals.json'

# Telegram configuration from environment variables
TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN', '8313486401:AAEtlDyCAvYkdF7tG1vyUQXual26ppA9MeM')
TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID', '6303104096')

# Rate limiting configuration
RATE_LIMIT_DELAY = 0.5
MAX_RETRIES = 3
RETRY_DELAY = 2.0

# Updated Symbol mapping
SYMBOL_MAP = {
    '1000PEPE/USDT': {'mexc':'PEPE1000/USDT','gateio':'PEPE1000/USDT','bitget':'PEPE1000/USDT'},
    '1000BONK/USDT': {'mexc':'BONK1000/USDT'},
    '1000FLOKI/USDT': {'mexc':'FLOKI1000/USDT'},
    '1000SHIB/USDT': {'mexc':'SHIB1000/USDT'},
    '1000CAT/USDT': {'mexc':'CAT1000/USDT'},
    'WIF/USDT': {'okx': 'WIF-USDT'},
    'NEIROETH/USDT': {'okx': 'NEIRO-USDT'}
}

# ---------------- logging setup ----------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# ---------------- globals ----------------
exchanges = {}
sr_cache = {}
loaded_model = None
USE_EXTERNAL_MODEL = False
prev_signals = {}

# ---------------- GitHub repository functions ----------------
async def read_file_from_github(filename):
    """Read file from GitHub repository"""
    if not GITHUB_TOKEN or not GITHUB_REPO:
        logger.warning(f"GitHub credentials missing, using local file: {filename}")
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            return None
        except Exception as e:
            logger.error(f"Error reading local file {filename}: {e}")
            return None

    try:
        url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{filename}"
        headers = {
            'Authorization': f'token {GITHUB_TOKEN}',
            'Accept': 'application/vnd.github.v3+json'
        }

        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    content = base64.b64decode(data['content']).decode('utf-8')
                    return content
                elif response.status == 404:
                    logger.info(f"File {filename} not found in repository, will create new")
                    return None
                else:
                    logger.error(f"Error reading {filename} from GitHub: {response.status}")
                    return None
    except Exception as e:
        logger.error(f"GitHub read error for {filename}: {e}")
        return None

async def write_file_to_github(filename, content, commit_message):
    """Write file to GitHub repository"""
    if not GITHUB_TOKEN or not GITHUB_REPO:
        logger.warning(f"GitHub credentials missing, saving locally: {filename}")
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        except Exception as e:
            logger.error(f"Error writing local file {filename}: {e}")
            return False

    try:
        # First get current file SHA if it exists
        get_url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{filename}"
        headers = {
            'Authorization': f'token {GITHUB_TOKEN}',
            'Accept': 'application/vnd.github.v3+json'
        }

        sha = None
        async with aiohttp.ClientSession() as session:
            async with session.get(get_url, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    sha = data['sha']

        # Prepare commit data
        encoded_content = base64.b64encode(content.encode('utf-8')).decode('utf-8')
        commit_data = {
            'message': commit_message,
            'content': encoded_content,
            'committer': {
                'name': GITHUB_ACTOR,
                'email': f'{GITHUB_ACTOR}@github.com'
            }
        }

        if sha:
            commit_data['sha'] = sha

        # Commit file
        put_url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{filename}"
        async with aiohttp.ClientSession() as session:
            async with session.put(put_url, headers=headers, json=commit_data) as response:
                if response.status in [200, 201]:
                    logger.info(f"Successfully committed {filename} to GitHub")
                    return True
                else:
                    logger.error(f"Error committing {filename} to GitHub: {response.status}")
                    response_text = await response.text()
                    logger.error(f"Response: {response_text}")
                    return False

    except Exception as e:
        logger.error(f"GitHub write error for {filename}: {e}")
        return False

# ---------------- Telegram functions ----------------
async def send_telegram_message(message):
    """Send message to Telegram chat"""
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {
            'chat_id': TELEGRAM_CHAT_ID,
            'text': message,
            'parse_mode': 'HTML'
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, timeout=10) as response:
                if response.status == 200:
                    return True
                else:
                    logger.error(f"Telegram send failed: {response.status}")
                    return False
    except Exception as e:
        logger.error(f"Telegram error: {e}")
        return False

async def send_error_notification(error_msg):
    """Send error notification to Telegram"""
    message = f"🚨 <b>SCANNER ERROR</b> 🚨\n\n"
    message += f"Error: {error_msg}\n"
    message += f"Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}"
    await send_telegram_message(message)

async def format_and_send_signals(confirmed_list):
    """Format confirmed signals and send to Telegram"""
    if not confirmed_list:
        return

    message = "🚨 <b>CONFIRMED TRADING SIGNALS</b> 🚨\n\n"

    for signal in confirmed_list:
        direction_emoji = "🟢" if signal['direction'] == 'LONG' else "🔴"
        price_fmt = f"{signal['price']:.8f}" if signal['price'] < 1 else f"{signal['price']:.6f}"

        message += f"{direction_emoji} <b>{signal['symbol']}</b>\n"
        message += f"Direction: <b>{signal['direction']}</b>\n"
        message += f"Exchange: {signal['exchange']}\n"
        message += f"Price: ${price_fmt}\n"
        message += f"Confidence: {signal['confidence']:.2f}\n"
        message += f"Count: {signal['count']}\n"
        message += f"RSI: {signal['rsi']:.2f}\n"
        message += "────────────────\n"

    message += f"\n⏰ Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}"
    message += f"\n🔄 Next scan in 15 minutes"

    await send_telegram_message(message)

# ---------------- persistence helpers ----------------
async def load_prev_signals():
    """Load previous signals from GitHub repository"""
    global prev_signals
    try:
        content = await read_file_from_github(PERSIST_PATH)
        if content:
            prev_signals = json.loads(content)
            logger.info(f"Loaded {len(prev_signals)} previous signals")
        else:
            prev_signals = {}
            logger.info("No previous signals found, starting fresh")
    except Exception as e:
        logger.error(f"Failed to load prev_signals: {e}")
        prev_signals = {}

async def save_prev_signals():
    """Save previous signals to GitHub repository"""
    try:
        content = json.dumps(prev_signals, ensure_ascii=False, indent=2)
        commit_message = f"Update signals memory - {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
        success = await write_file_to_github(PERSIST_PATH, content, commit_message)
        if success:
            logger.info(f"Saved {len(prev_signals)} signals to repository")
        return success
    except Exception as e:
        logger.error(f"Failed to save prev_signals: {e}")
        return False

# ---------------- CSV logging ----------------
async def append_csv_row(row):
    """Append row to CSV log in GitHub repository"""
    header = ['datetime_utc','symbol','direction','exchange','confidence','price','rsi','count']

    try:
        # Read existing CSV content
        existing_content = await read_file_from_github(CSV_LOG_PATH)

        # Prepare new row
        new_row = ','.join(str(row.get(col,'')) for col in header)

        if existing_content:
            # Append to existing content
            updated_content = existing_content.rstrip() + '\n' + new_row
        else:
            # Create new file with header
            updated_content = ','.join(header) + '\n' + new_row

        # Save back to repository
        commit_message = f"Add confirmed signal: {row.get('symbol', 'UNKNOWN')} {row.get('direction', '')}"
        success = await write_file_to_github(CSV_LOG_PATH, updated_content, commit_message)

        if success:
            logger.info(f"Logged signal to CSV: {row.get('symbol')} {row.get('direction')}")

        return success

    except Exception as e:
        logger.error(f"Failed to append CSV: {e}")
        return False

# ---------------- exchange helpers ----------------
def get_symbol_for_exchange(symbol, exchange_id):
    """Get the correct symbol format for a specific exchange"""
    if symbol in SYMBOL_MAP and exchange_id in SYMBOL_MAP[symbol]:
        return SYMBOL_MAP[symbol][exchange_id]
    return symbol

async def create_exchange_client(exchange_id):
    try:
        cls = getattr(ccxt, exchange_id)
        client = cls({
            'enableRateLimit': True,
            'timeout': 30000,
            'rateLimit': 1200,
        })
        try:
            await client.load_markets()
        except Exception:
            pass
        return client
    except Exception as e:
        logger.error(f"Init exchange {exchange_id} failed: {e}")
        return None

async def init_exchanges():
    for eid in EXCHANGE_ORDER:
        client = await create_exchange_client(eid)
        if client:
            exchanges[eid] = client
            logger.info(f"Initialized exchange: {eid}")
        await asyncio.sleep(0.5)

    logger.info(f"Total exchanges initialized: {len(exchanges)}")

async def close_exchanges():
    for c in exchanges.values():
        try:
            await c.close()
        except Exception:
            pass

# ---------------- market data & indicators ----------------
async def fetch_ohlcv_with_retry(exchange, symbol, timeframe='15m', limit=300):
    """Fetch OHLCV with retry logic and rate limit handling"""
    for attempt in range(MAX_RETRIES):
        try:
            mapped = get_symbol_for_exchange(symbol, exchange.id)
            await asyncio.sleep(RATE_LIMIT_DELAY)

            ohlcv = await exchange.fetch_ohlcv(mapped, timeframe=timeframe, limit=limit)

            if not ohlcv or len(ohlcv) < 20:
                return None

            df = pd.DataFrame(ohlcv, columns=['timestamp','open','high','low','close','volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
            df = df.sort_values('timestamp').reset_index(drop=True)
            return df

        except Exception as e:
            error_str = str(e).lower()

            if any(phrase in error_str for phrase in ['rate limit', 'too many', 'exceeded', '429']):
                if attempt < MAX_RETRIES - 1:
                    delay = RETRY_DELAY * (2 ** attempt)
                    logger.warning(f"Rate limit hit for {exchange.id} {symbol}, retrying in {delay}s")
                    await asyncio.sleep(delay)
                    continue
                else:
                    logger.error(f"Rate limit exceeded after {MAX_RETRIES} attempts: {exchange.id} {symbol}")
                    return None

            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(RETRY_DELAY)
                continue
            else:
                logger.debug(f"fetch_ohlcv error {getattr(exchange,'id', 'unknown')} {symbol}: {e}")
                return None

    return None

def calculate_indicators(df):
    if df is None or len(df) < 20:
        return df

    def rsi(series, period=14):
        delta = series.diff()
        gain = delta.where(delta>0,0).rolling(window=period).mean()
        loss = (-delta).where(delta<0,0).rolling(window=period).mean()
        rs = gain/(loss+1e-10)
        return 100 - (100/(1+rs))

    def wavetrend(df_in, n1=10, n2=11):
        ap = (df_in['high'] + df_in['low'] + df_in['close'])/3
        esa = ap.ewm(span=n1).mean()
        d = (ap - esa).abs().ewm(span=n1).mean()
        ci = (ap - esa)/(0.015*d + 1e-10)
        return ci.ewm(span=n2).mean()

    def cci(df_in, period=20):
        tp = (df_in['high'] + df_in['low'] + df_in['close'])/3
        sma = tp.rolling(window=period).mean()
        mad = tp.rolling(window=period).apply(lambda x: np.abs(x - x.mean()).mean())
        return (tp - sma)/(0.015*mad + 1e-10)

    def adx(df_in, period=20):
        high, low, close = df_in['high'], df_in['low'], df_in['close']
        up = high.diff()
        down = -low.diff()
        plus_dm = up.where((up>down) & (up>0), 0)
        minus_dm = down.where((down>up) & (down>0), 0)
        tr = pd.concat([high-low, (high-close.shift(1)).abs(), (low-close.shift(1)).abs()], axis=1).max(axis=1)
        plus_di = (plus_dm.rolling(window=period).mean()/(tr.rolling(window=period).mean()+1e-10))*100
        minus_di = (minus_dm.rolling(window=period).mean()/(tr.rolling(window=period).mean()+1e-10))*100
        dx = (plus_di - minus_di).abs()/(plus_di+minus_di+1e-10)*100
        return dx.rolling(window=period).mean()

    try:
        df['rsi'] = rsi(df['close'], 14)
        df['rsi_short'] = rsi(df['close'], 9)
        df['wavetrend'] = wavetrend(df)
        df['cci'] = cci(df)
        df['adx'] = adx(df)
    except Exception as e:
        logger.error(f"Indicator calc failed: {e}")
        for col in ['rsi','rsi_short','wavetrend','cci','adx']:
            df[col] = 0

    return df

def normalize_features(df):
    cols = ['rsi','wavetrend','cci','adx','rsi_short']
    for c in cols:
        if c in df.columns:
            std = df[c].std()
            if std > 1e-6:
                df[c+'_norm'] = np.tanh(df[c]/std)
            else:
                df[c+'_norm'] = 0
    return df

# ---------------- S/R Analysis (keeping key functions) ----------------
def detect_pivots(highs, lows, window=PIVOT_WINDOW):
    ph, pl = [], []
    for i in range(window, len(highs)-window):
        if all(highs[i] >= highs[j] for j in range(i-window, i+window+1) if j!=i) and highs[i]==max(highs[i-window:i+window+1]):
            ph.append((i, highs[i]))
        if all(lows[i] <= lows[j] for j in range(i-window, i+window+1) if j!=i) and lows[i]==min(lows[i-window:i+window+1]):
            pl.append((i, lows[i]))
    return ph, pl

def cluster_levels(levels, atr_value, tolerance_multiplier=SR_TOLERANCE_ATR):
    if not levels or atr_value <= 0:
        return []

    tol = atr_value * tolerance_multiplier
    levels = sorted(levels, key=lambda x: x[1])
    clusters = []
    cur = [levels[0]]

    for lvl in levels[1:]:
        if abs(lvl[1] - cur[-1][1]) <= tol:
            cur.append(lvl)
        else:
            avg = np.mean([x[1] for x in cur])
            clusters.append((avg, len(cur)))
            cur = [lvl]

    if cur:
        avg = np.mean([x[1] for x in cur])
        clusters.append((avg, len(cur)))

    return clusters

def calculate_sr_levels(df, timeframe):
    if df is None or len(df) < 20:
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
    valid_resist = [(p, touches*weight, timeframe) for p, touches in resist if touches >= MIN_TOUCHES]
    valid_support = [(p, touches*weight, timeframe) for p, touches in support if touches >= MIN_TOUCHES]

    return valid_resist, valid_support

async def get_multi_timeframe_sr(exchange, symbol):
    all_r, all_s = [], []

    for tf in SR_TIMEFRAMES:
        limit_map = {'1d':100,'4h':200,'1h':250,'15m':300,'5m':300}
        limit = limit_map.get(tf, 300)

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
        if nearby:
            nearest = min(nearby, key=lambda x:x[0])
            if (nearest[0] - current_price) < threshold:
                return False
    else:
        nearby = [s for s in support_levels if s[0] < current_price]
        if nearby:
            nearest = max(nearby, key=lambda x:x[0])
            if (current_price - nearest[0]) < threshold:
                return False

    return True

# ---------------- ML prediction ----------------
def create_labels(df, forward_bars=4, threshold=0.002):
    returns = df['close'].shift(-forward_bars)/df['close'] - 1
    labels = np.where(returns >= threshold, 1, np.where(returns <= -threshold, -1, 0))
    return labels

def lorentzian_knn_predict(df, k=8):
    feature_cols = ['rsi_norm','wavetrend_norm','cci_norm','adx_norm','rsi_short_norm']

    if df is None or len(df) < 50 or not all(c in df.columns for c in feature_cols):
        return 0, 0

    labels = create_labels(df)
    recent_df = df.iloc[-200:].copy()
    recent_labels = labels[-200:]

    if len(recent_df) < k:
        return 0, 0

    current = recent_df[feature_cols].iloc[-1].values
    distances, valid_idx = [], []

    for i in range(0, len(recent_df)-5, 4):
        if pd.notna(recent_labels[i]) and recent_labels[i] != 0:
            hist = recent_df[feature_cols].iloc[i].values
            if not np.any(np.isnan(hist)) and not np.any(np.isnan(current)):
                d = np.sum(np.log(1 + np.abs(current - hist)))
                distances.append(d); valid_idx.append(i)

    if len(distances) < k:
        return 0,0

    idxs = np.argsort(distances)[:k]
    neighbor_labels = [recent_labels[valid_idx[i]] for i in idxs]

    prediction_sum = sum(neighbor_labels)
    confidence = abs(np.mean(neighbor_labels))

    return prediction_sum, confidence

# ---------------- Main scanning logic ----------------
def _base_quote_from_unified(sym: str):
    try:
        base, right = sym.split('/')
        quote = right.split(':')[0]
        return base, quote
    except Exception:
        return sym, ''

async def _ensure_swap_options(exchange, eid: str):
    try:
        opts = getattr(exchange, 'options', {}) or {}
        if opts.get('defaultType') != 'swap':
            exchange.options = {**opts, 'defaultType': 'swap'}
            try:
                await exchange.load_markets()
            except Exception:
                pass
    except Exception:
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

        if '/' in requested_symbol:
            b2, q2 = _base_quote_from_unified(requested_symbol)
            colon2 = f"{b2}/{q2}:{q2}"
            if colon2 in markets and markets[colon2].get('swap'):
                return colon2

    except Exception:
        return None

    return None

def _pick_supported_timeframe(exchange):
    tfs = getattr(exchange, 'timeframes', {}) or {}
    if not tfs:
        return '15m'

    preferred_tfs = ['15m', '5m', '1h', '4h', '1d']
    for tf in preferred_tfs:
        if tf in tfs:
            return tf

    return next(iter(tfs.keys()), '15m')

async def scan_market_single_run():
    """Single market scan execution for GitHub Actions"""
    logger.info("Starting market scan...")

    # Clear cache for fresh start
    sr_cache.clear()

    # Ensure all exchanges are in swap mode
    for eid, ex in exchanges.items():
        await _ensure_swap_options(ex, eid)

    current_map = {}
    confirmed_list = []
    now_iso = datetime.now(timezone.utc).isoformat()

    # Process each coin across exchanges
    for i, requested_symbol in enumerate(COIN_LIST):
        logger.info(f"Processing {requested_symbol} ({i+1}/{len(COIN_LIST)})")
        found = False

        for eid in EXCHANGE_ORDER:
            exchange = exchanges.get(eid)
            if not exchange:
                continue

            try:
                resolved = _resolve_swap_symbol(exchange, eid, requested_symbol)
            except Exception:
                resolved = None

            if not resolved:
                continue

            tf = _pick_supported_timeframe(exchange)
            if '15m' in getattr(exchange, 'timeframes', {}):
                tf = '15m'

            df = await fetch_ohlcv_with_retry(exchange, resolved, timeframe=tf, limit=300)

            if df is None:
                continue

            # Technical analysis
            df = calculate_indicators(df)
            df = normalize_features(df)

            # ML prediction
            prediction_sum, confidence = lorentzian_knn_predict(df)

            if confidence < 0.6 or abs(prediction_sum) < 2:
                continue

            current_price = float(df['close'].iloc[-1])
            current_rsi = float(df['rsi'].iloc[-1]) if 'rsi' in df.columns else float('nan')

            signal_type = 'LONG' if prediction_sum > 0 else 'SHORT'
            rsi_agrees = (current_rsi < 70) if signal_type == 'LONG' else (current_rsi > 30)

            if not rsi_agrees:
                continue

            # S/R analysis
            cache_key = f"{requested_symbol}_{eid}"
            if cache_key not in sr_cache:
                resist, support = await get_multi_timeframe_sr(exchange, resolved)
                sr_cache[cache_key] = (resist, support)
            else:
                resist, support = sr_cache[cache_key]

            df['tr'] = np.maximum(
                df['high'] - df['low'],
                np.maximum((df['high'] - df['close'].shift(1)).abs(), (df['low'] - df['close'].shift(1)).abs())
            )

            atr = df['tr'].rolling(14).mean().iloc[-1]
            if pd.isna(atr):
                atr = (df['high'] - df['low']).mean()

            if not check_sr_proximity(current_price, resist, support, float(atr), signal_type):
                continue

            current_map[requested_symbol] = {
                'symbol': requested_symbol,
                'exchange': eid,
                'direction': signal_type,
                'rsi': current_rsi,
                'confidence': float(confidence),
                'price': current_price,
                'ts': now_iso
            }

            found = True
            logger.info(f"Signal found: {requested_symbol} {signal_type} on {eid}")
            break

        await asyncio.sleep(0.1)  # Small delay between coins

    logger.info(f"Found {len(current_map)} potential signals")

    # Confirmation logic
    for sym, cur in current_map.items():
        prev = prev_signals.get(sym)

        if prev and prev.get('direction') == cur['direction']:
            new_count = prev.get('count', 1) + 1
            prev_signals[sym] = {
                'direction': cur['direction'],
                'confidence': cur['confidence'],
                'count': new_count,
                'exchange': cur['exchange'],
                'price': cur['price'],
                'last_seen': cur['ts']
            }
        else:
            prev_signals[sym] = {
                'direction': cur['direction'],
                'confidence': cur['confidence'],
                'count': 1,
                'exchange': cur['exchange'],
                'price': cur['price'],
                'last_seen': cur['ts']
            }

    # Generate confirmed signals list
    for sym, data in prev_signals.items():
        if sym in current_map and data.get('count', 0) >= 2:
            confirmed_list.append({
                'symbol': sym,
                'exchange': data['exchange'],
                'direction': data['direction'],
                'rsi': current_map[sym].get('rsi', float('nan')),
                'confidence': data['confidence'],
                'price': data['price'],
                'count': data['count']
            })

            # Log to CSV
            csv_row = {
                'datetime_utc': now_iso,
                'symbol': sym,
                'direction': data['direction'],
                'exchange': data['exchange'],
                'confidence': data['confidence'],
                'price': data['price'],
                'rsi': current_map[sym].get('rsi', ''),
                'count': data['count']
            }

            await append_csv_row(csv_row)

    # Remove signals not seen in current scan
    missing = [s for s in list(prev_signals.keys()) if s not in current_map]
    for s in missing:
        prev_signals.pop(s, None)

    # Save persistence data
    await save_prev_signals()

    # Output results
    if confirmed_list:
        logger.info(f"CONFIRMED SIGNALS: {len(confirmed_list)}")
        for signal in confirmed_list:
            logger.info(f"  {signal['symbol']} {signal['direction']} {signal['exchange']} (Count: {signal['count']})")

        # Send Telegram alerts
        await format_and_send_signals(confirmed_list)
    else:
        logger.info("No confirmed signals found")

    logger.info("Market scan completed")
    return confirmed_list

# ---------------- Model loader ----------------
def try_load_model():
    global loaded_model, USE_EXTERNAL_MODEL

    if joblib is None:
        USE_EXTERNAL_MODEL = False
        return

    if os.path.exists(MODEL_PATH):
        try:
            loaded_model = joblib.load(MODEL_PATH)
            USE_EXTERNAL_MODEL = True
            logger.info("External model loaded successfully")
        except Exception as e:
            logger.error(f"Failed loading model: {e}")
            USE_EXTERNAL_MODEL = False
    else:
        USE_EXTERNAL_MODEL = False
        logger.info("No external model found, using internal k-NN")

# ---------------- Main entry point ----------------
async def main():
    """Main function for single GitHub Actions execution"""
    try:
        start_time = time.time()
        logger.info("=" * 60)
        logger.info("GITHUB ACTIONS CRYPTO SCANNER STARTING")
        logger.info("=" * 60)
        logger.info(f"Scan time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
        logger.info(f"Coins to analyze: {len(COIN_LIST)}")
        logger.info(f"Exchanges: {', '.join(EXCHANGE_ORDER)}")

        # Load persistence data
        await load_prev_signals()

        # Try to load external model
        try_load_model()

        # Initialize exchanges
        await init_exchanges()

        if not exchanges:
            raise Exception("No exchanges initialized successfully")

        # Run single market scan
        confirmed_signals = await scan_market_single_run()

        # Cleanup
        await close_exchanges()

        elapsed = time.time() - start_time
        logger.info(f"Scan completed in {elapsed:.2f} seconds")
        logger.info(f"Results: {len(confirmed_signals)} confirmed signals")
        logger.info("=" * 60)

        return len(confirmed_signals)

    except Exception as e:
        error_msg = f"Scanner failed: {str(e)}"
        logger.error(error_msg)
        await send_error_notification(error_msg)
        raise

if __name__ == '__main__':
    try:
        result = asyncio.run(main())
        sys.exit(0)  # Success
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)  # Failure
