import yfinance as yf
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
import warnings
import requests
import time
import os
import sys
import datetime

td = datetime.date.today()
today = td.strftime('%Y-%m-%d')
warnings.filterwarnings("ignore")


# ----------------------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------------------
BACKTEST = True
INITIAL_CAPITAL = 100_000
SLIPPAGE_PCT = 0.0005
COMMISSION_PER_TRADE = 0.0
MAX_SECTOR_ALLOCATION = 0.25
MIN_MARKET_CAP = 10_000_000      # $10M
MIN_AVG_VOLUME = 500_000         # Lowered for more signals
MIN_PRICE = 5.0
LOOKBACK_DAYS = 252
RSI_WINDOW = 14
SMA_WINDOW = 20
BB_WINDOW = 20
BB_STD = 2

OUTPUT_DIR = r"C:\Users\User\Desktop\Trade File"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ----------------------------------------------------------------------
# 1. Get S&P 500 + Metadata (NO SESSION INJECTION)
# ----------------------------------------------------------------------
def get_sp500_universe():
    print("Fetching S&P 500 tickers from Wikipedia...")
    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
        df = pd.read_html(resp.text)[0]
        df['Symbol'] = df['Symbol'].str.replace('.', '-')
        tickers = df['Symbol'].tolist()[:200]  # Top 200
        print(f"Found {len(tickers)} tickers. Fetching metadata...")
    except Exception as e:
        print(f"Wikipedia failed: {e}. Using minimal fallback.")
        return get_minimal_fallback()

    # Use yf.Tickers() for batch metadata (no session injection)
    print("Downloading metadata in batch...")
    try:
        ticker_objects = yf.Tickers(' '.join(tickers))
        info_data = []
        for t in tickers:
            try:
                info = ticker_objects.tickers[t].info
                market_cap = info.get('marketCap', 0) or 0
                sector = info.get('sector', 'Unknown')
                if market_cap >= MIN_MARKET_CAP:
                    info_data.append({
                        'Symbol': t,
                        'Sector': sector,
                        'MarketCap': market_cap
                    })
            except:
                continue
            time.sleep(0.5)  # Respect rate limits
        universe = pd.DataFrame(info_data)
        print(f"Universe loaded: {len(universe)} stocks ≥ $10M")
        return universe if len(universe) > 0 else get_minimal_fallback()
    except Exception as e:
        print(f"yfinance batch failed: {e}. Using fallback.")
        return get_minimal_fallback()


def get_minimal_fallback():
    print("Using minimal 10-stock fallback...")
    return pd.DataFrame([
        {'Symbol': 'AAPL', 'Sector': 'Technology', 'MarketCap': 2.8e12},
        {'Symbol': 'MSFT', 'Sector': 'Technology', 'MarketCap': 3.1e12},
        {'Symbol': 'NVDA', 'Sector': 'Technology', 'MarketCap': 2.2e12},
        {'Symbol': 'GOOGL', 'Sector': 'Communication Services', 'MarketCap': 1.9e12},
        {'Symbol': 'AMZN', 'Sector': 'Consumer Discretionary', 'MarketCap': 1.8e12},
        {'Symbol': 'META', 'Sector': 'Communication Services', 'MarketCap': 1.2e12},
        {'Symbol': 'TSLA', 'Sector': 'Consumer Discretionary', 'MarketCap': 1.1e12},
        {'Symbol': 'BRK-B', 'Sector': 'Financials', 'MarketCap': 9.5e11},
        {'Symbol': 'JPM', 'Sector': 'Financials', 'MarketCap': 5.8e11},
        {'Symbol': 'JNJ', 'Sector': 'Healthcare', 'MarketCap': 4.2e11},
    ])


# ----------------------------------------------------------------------
# 2. Safe Download (No Session, Retry)
# ----------------------------------------------------------------------
def safe_download(tickers):
    print(f"Downloading OHLC for {len(tickers)} stocks...")
    for attempt in range(3):
        try:
            data = yf.download(tickers, period="2y", progress=False, threads=False)
            if not data.empty and 'Close' in data.columns:
                print(f"Download successful: {len(data.columns.levels[1] if isinstance(data.columns, pd.MultiIndex) else len(data.columns))} tickers")
                return data
        except Exception as e:
            print(f"Download attempt {attempt+1} failed: {e}")
        time.sleep(5)
    print("All downloads failed.")
    sys.exit(1)


# ----------------------------------------------------------------------
# 3. Indicators & Signals
# ----------------------------------------------------------------------
def adf_decisive_test(series):
    series = series.dropna()
    if len(series) < 50: return False, np.nan, False, "Too few"
    res = adfuller(series, autolag="AIC")
    p = res[1]
    if p <= 0.01: return True, p, True, "Stationary"
    if p >= 0.10: return False, p, True, "Non-Stationary"
    return (p <= 0.05), p, False, f"Indecisive {p:.3f}"

def calculate_rsi(series, window=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(window).mean()
    loss = -delta.where(delta < 0, 0).rolling(window).mean()
    loss = loss.replace(0, np.nan)
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return round(rsi.iloc[-1], 2) if len(rsi) > 0 else 100

def calculate_sma(series, window=20):
    return round(series.rolling(window).mean().iloc[-1], 2)

def calculate_bollinger(series, window=20, std_dev=2):
    sma = series.rolling(window).mean()
    std = series.rolling(window).std()
    lower = sma - std_dev * std
    upper = sma + std_dev * std
    return round(lower.iloc[-1], 2), round(sma.iloc[-1], 2), round(upper.iloc[-1], 2)

def momentum_signal(rsi, close, sma20):
    return "BUY" if rsi < 35 and close > sma20 else "HOLD"

def mean_reversion_signal(rsi, close, lower_bb, upper_bb):
    return "BUY" if close < lower_bb and rsi < 35 else "HOLD"


# ----------------------------------------------------------------------
# 4. Backtest
# ----------------------------------------------------------------------
def run_backtest(signals_df, ohlc_data):
    capital = INITIAL_CAPITAL
    daily_pnl = []
    trade_log = []

    for i in range(len(ohlc_data) - 1):
        date = ohlc_data.index[i]
        next_date = ohlc_data.index[i + 1]
        day_signals = signals_df[signals_df['Date'] == date.strftime('%Y-%m-%d')]
        if day_signals.empty:
            daily_pnl.append(0)
            continue

        pnl = 0
        for _, sig in day_signals.iterrows():
            t = sig['Ticker']
            if t not in ohlc_data.columns.get_level_values(1):
                continue
            try:
                open_price = ohlc_data.loc[next_date, ('Open', t)]
                close_price = ohlc_data.loc[next_date, ('Close', t)]
            except:
                continue
            if pd.isna(open_price) or pd.isna(close_price):
                continue

            alloc = sig['Allocation_%'] / 100
            dollar = capital * alloc
            buy_price = open_price * (1 + SLIPPAGE_PCT)
            shares = int(dollar // buy_price)
            if shares == 0: continue

            cost = shares * buy_price + COMMISSION_PER_TRADE
            sell_price = close_price * (1 - SLIPPAGE_PCT)
            proceeds = shares * sell_price - COMMISSION_PER_TRADE
            trade_pnl = proceeds - cost
            pnl += trade_pnl

            trade_log.append({
                'Date': next_date.strftime('%Y-%m-%d'),
                'Ticker': t,
                'Shares': shares,
                'Buy': round(buy_price, 2),
                'Sell': round(sell_price, 2),
                'PnL': round(trade_pnl, 2)
            })

        capital += pnl
        daily_pnl.append(pnl)

    total_return = (capital - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
    win_rate = sum(1 for p in daily_pnl if p > 0) / len([p for p in daily_pnl if p != 0]) * 100 if any(daily_pnl) else 0

    print(f"\nBACKTEST RESULTS")
    print(f"Start: ${INITIAL_CAPITAL:,.0f} → End: ${capital:,.0f} ({total_return:+.2f}%)")
    print(f"Win Rate: {win_rate:.1f}% | Trades: {len(trade_log)}")

    pd.DataFrame(trade_log).to_csv(os.path.join(OUTPUT_DIR, f"daytrade_log_{today}.csv"), index=False)
    pd.DataFrame({'Daily_PnL': daily_pnl}).to_csv(os.path.join(OUTPUT_DIR, f"daily_pnl_{today}.csv"), index=False)
    print(f"Logs saved to: {OUTPUT_DIR}")


# ----------------------------------------------------------------------
# 5. Main
# ----------------------------------------------------------------------
def main():
    universe = get_sp500_universe()
    tickers = universe['Symbol'].tolist()
    print(f"Scanning {len(tickers)} stocks for day-trade signals...\n")

    data = safe_download(tickers)
    close = data['Close']
    open_p = data['Open']
    volume = data['Volume']

    recent_close = close.tail(LOOKBACK_DAYS + 1)
    recent_open = open_p.tail(LOOKBACK_DAYS + 1)
    recent_vol = volume.tail(LOOKBACK_DAYS + 1)

    signals = []
    for ticker in tickers:
        try:
            c = recent_close[ticker].dropna()
            v = recent_vol[ticker].dropna()
            if len(c) < 100 or v[-60:].mean() < MIN_AVG_VOLUME:
                continue

            row = universe[universe['Symbol'] == ticker].iloc[0]
            signal_date = c.index[-2]
            price = c.loc[signal_date]
            if price < MIN_PRICE:
                continue

            log_ret = np.log(c).diff().dropna()
            if len(log_ret) < 50:
                continue

            stationary, pval, decisive, _ = adf_decisive_test(log_ret)
            if not decisive:
                continue

            hist = c.loc[:signal_date]
            rsi = calculate_rsi(hist)
            sma20 = calculate_sma(hist)
            lower_bb, _, upper_bb = calculate_bollinger(hist)

            sig = "HOLD"
            if not stationary:
                sig = momentum_signal(rsi, price, sma20)
            else:
                sig = mean_reversion_signal(rsi, price, lower_bb, upper_bb)

            if sig == "BUY":
                signals.append({
                    'Date': signal_date.strftime('%Y-%m-%d'),
                    'Ticker': ticker,
                    'Close': price,
                    'p_value': pval,
                    'Stationary': stationary,
                    'Sector': row['Sector']
                })
        except:
            continue

    if not signals:
        print("No signals found. Try lowering MIN_AVG_VOLUME or running after market close.")
        return

    df = pd.DataFrame(signals)
    stat_df = df[df['Stationary']].copy()
    trend_df = df[~df['Stationary']].copy()
    if len(stat_df) > 0:
        stat_df['Confidence'] = 1 / np.maximum(stat_df['p_value'], 1e-6)
    if len(trend_df) > 0:
        trend_df['Confidence'] = trend_df['p_value']
    df = pd.concat([stat_df, trend_df], ignore_index=True)

    total = df['Confidence'].sum()
    df['Raw_Alloc'] = df['Confidence']  
    sector_sum = df.groupby('Sector')['Raw_Alloc'].sum()
    for sector in sector_sum[sector_sum > MAX_SECTOR_ALLOCATION].index:
        scale = MAX_SECTOR_ALLOCATION / sector_sum[sector]
        df.loc[df['Sector'] == sector, 'Raw_Alloc'] *= scale
    df['Raw_Alloc'] = df['Raw_Alloc'] / df['Raw_Alloc'].sum()
    df['Allocation_%'] = (df['Raw_Alloc'] * 100).round(2)

    trade_file = df[['Date', 'Ticker', 'Close', 'Allocation_%', 'Sector']].sort_values(['Date', 'Allocation_%'], ascending=[False, False])
    path = os.path.join(OUTPUT_DIR, f"daytrade_signals_{today}.csv")
    trade_file.to_csv(path, index=False)
    print(f"\nSignals saved: {path}")
    print(trade_file.head(10).to_string(index=False))

    if BACKTEST:
        ohlc = pd.concat([open_p, close], axis=1, keys=['Open', 'Close'])
        ohlc = ohlc.swaplevel(axis=1).sort_index(axis=1)
        run_backtest(trade_file, ohlc.tail(LOOKBACK_DAYS))


if __name__ == "__main__":
    main()