# 🚀 Crypto Trading Signal Scanner

Automated cryptocurrency trading signal scanner that runs on GitHub Actions every 15 minutes.

## 📊 Features

- **40 Cryptocurrency Pairs**: Comprehensive meme coin and major altcoin coverage
- **8 Exchange Support**: Binance, Bybit, KuCoin, MEXC, Gate.io, OKX, Bitget, Huobi
- **15-minute Analysis**: Runs every 15 minutes on 15m timeframes
- **Multi-timeframe S/R**: 1d, 4h, 1h, 15m, 5m support/resistance analysis
- **ML Predictions**: Lorentzian k-NN algorithm for signal generation
- **Persistent Memory**: Confirmed signals after 2+ consecutive detections
- **Telegram Alerts**: Instant notifications for confirmed signals
- **GitHub Storage**: Free persistence using repository as database
- **Rate Limit Protection**: Smart retry logic and exponential backoff

## 🎯 Monitored Coins (40 pairs)

```
DOGE, SHIB, PEPE, WIF, BONK, FLOKI, MEME, KOMA, DOGS, NEIROETH,
1000RATS, ORDI, PIPPIN, BAN, 1000SHIB, OM, CHILLGUY, PONKE, BOME,
MYRO, PEOPLE, PENGU, SPX, 1000BONK, PNUT, FARTCOIN, HIPPO, AIXBT,
BRETT, VINE, MOODENG, MUBARAK, MEW, POPCAT, 1000FLOKI, 1000CAT,
ACT, SLERF, DEGEN, 1000PEPE
```

## 🔧 Setup Instructions

### 1. Fork this Repository
- Click "Fork" at the top of this repository
- Create your own copy

### 2. Set up GitHub Secrets
Go to your repository Settings → Secrets and variables → Actions

Add these secrets:
```
TELEGRAM_BOT_TOKEN: Your Telegram bot token
TELEGRAM_CHAT_ID: Your Telegram chat ID
```

### 3. Enable GitHub Actions
- Go to the "Actions" tab in your repository
- Click "I understand my workflows, go ahead and enable them"

### 4. Manual Test Run (Optional)
- Go to Actions → "Crypto Trading Scanner"
- Click "Run workflow" → "Run workflow"

## 📅 Automatic Schedule

The scanner runs automatically every 15 minutes:
- **Cron Schedule**: `*/15 * * * *`
- **Timezone**: UTC
- **Runtime**: ~2-5 minutes per scan

## 📊 Signal Logic

### Technical Analysis
- **RSI**: 14-period momentum indicator
- **Wave Trend**: Custom oscillator
- **CCI**: Commodity Channel Index  
- **ADX**: Average Directional Index
- **Multi-timeframe S/R**: Support/resistance levels

### Signal Confirmation
1. **First Detection**: Signal stored in memory
2. **Second Detection**: Same signal → Telegram alert sent
3. **Persistence**: Memory maintained across GitHub Actions runs

### ML Algorithm
- **Lorentzian k-NN**: Distance-based similarity matching
- **Features**: Normalized technical indicators
- **Confidence Threshold**: 60%+ required
- **Prediction**: Long/Short signal generation

## 📱 Telegram Notifications

Example alert format:
```
🚨 CONFIRMED TRADING SIGNALS 🚨

🟢 DOGE/USDT
Direction: LONG
Exchange: binance
Price: $0.123456
Confidence: 0.85
Count: 2
RSI: 45.32
────────────────

⏰ Time: 2025-10-09 12:00:00 UTC
🔄 Next scan in 15 minutes
```

## 📈 Data Storage

All data is stored in the GitHub repository:

### Files Updated Automatically:
- `prev_signals.json` - Signal memory and confirmation counts
- `confirmed_signals_log.csv` - Historical log of all confirmed signals

### File Structure:
```
your-repo/
├── .github/workflows/scanner.yml    # GitHub Actions workflow
├── scanner.py                       # Main scanner code
├── requirements.txt                 # Python dependencies
├── prev_signals.json               # Signal persistence (auto-updated)
├── confirmed_signals_log.csv       # Signal history (auto-updated)
└── README.md                       # This file
```

## 🔍 Exchange Symbol Mapping

The scanner handles different symbol formats across exchanges:

```python
SYMBOL_MAP = {
    '1000PEPE/USDT': {'mexc':'PEPE1000/USDT', 'gateio':'PEPE1000/USDT', 'bitget':'PEPE1000/USDT'},
    '1000BONK/USDT': {'mexc':'BONK1000/USDT'},
    '1000FLOKI/USDT': {'mexc':'FLOKI1000/USDT'},
    '1000SHIB/USDT': {'mexc':'SHIB1000/USDT'},
    '1000CAT/USDT': {'mexc':'CAT1000/USDT'},
    'WIF/USDT': {'okx': 'WIF-USDT'},
    'NEIROETH/USDT': {'okx': 'NEIRO-USDT'}
}
```

## ⚙️ Configuration

### Key Parameters:
```python
SCAN_INTERVAL = "Every 15 minutes"
TIMEFRAME = "15m"
CONFIRMATION_REQUIRED = 2  # signals needed for alert
CONFIDENCE_THRESHOLD = 0.6  # 60%+ confidence required
RSI_LONG_THRESHOLD = 70    # RSI < 70 for long signals
RSI_SHORT_THRESHOLD = 30   # RSI > 30 for short signals
```

## 🛠️ Troubleshooting

### Common Issues:

1. **No Telegram alerts**: Check bot token and chat ID in secrets
2. **GitHub Actions failing**: Check if Actions are enabled in repository settings
3. **Exchange errors**: Rate limits are handled automatically with retries
4. **File commit errors**: Ensure GitHub token has write permissions

### Logs Location:
- GitHub Actions → Latest workflow run → View logs
- Error notifications sent to Telegram automatically

## 📊 Performance Stats

- **Scan Time**: ~2-5 minutes for 40 coins across 8 exchanges
- **API Calls**: ~320 per scan (rate limited)
- **Memory Usage**: < 100MB
- **GitHub Actions Limit**: 2000 minutes/month (free tier)
- **Estimated Usage**: ~150 minutes/month

## ⚠️ Disclaimers

- **Not Financial Advice**: These are technical analysis signals only
- **Risk Warning**: Cryptocurrency trading involves significant risk
- **Backtesting**: No guarantees of future performance
- **API Limits**: Subject to exchange rate limits and availability

## 🔄 Updates and Maintenance

The scanner is designed to be self-maintaining:
- ✅ Auto-updates persistence files
- ✅ Handles exchange outages gracefully  
- ✅ Retries failed API calls
- ✅ Sends error notifications via Telegram
- ✅ Logs detailed information for debugging

---

## 📞 Support

For issues or questions:
1. Check GitHub Actions logs
2. Review Telegram error notifications
3. Create GitHub issue in repository

**Happy Trading! 🚀**
