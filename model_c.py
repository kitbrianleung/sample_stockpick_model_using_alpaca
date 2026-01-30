import yfinance as yf
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
import warnings
import requests
import time
import os
import sys
from datetime import datetime
from alpaca_trade_api import REST
from alpaca_trade_api.rest import APIError

warnings.filterwarnings("ignore")

BACKTEST = False
LIVE_TRADE = True

# Alpaca API (Paper Trading)
ALPACA_API_KEY = "YOUR_ALPACA_PAPER_KEY"
ALPACA_SECRET_KEY = "YOUR_ALPACA_PAPER_SECRET"
ALPACA_BASE_URL = "https://paper-api.alpaca.markets"

# Parameters
SLIPPAGE_PCT = 0.0005
STOP_LOSS_PCT = 0.10        # 10% stop-loss
TAKE_PROFIT_PCT = 0.25      # 25% take-profit (0 = disabled)
MAX_SECTOR_ALLOCATION = 0.25
MIN_MARKET_CAP = 10_000_000
MIN_AVG_VOLUME = 500_000
MIN_PRICE = 5.0
LOOKBACK_DAYS = 252
RSI_WINDOW = 14
SMA_WINDOW = 20
BB_WINDOW = 20
BB_STD = 2

OUTPUT_DIR = r"C:\Users\User\Desktop\Trade File"
os.makedirs(OUTPUT_DIR, exist_ok=True)
SIGNALS_FILE = os.path.join(OUTPUT_DIR, "daytrade_signals.csv")



def get_alpaca_client():
    if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
        print("Alpaca keys missing.")
        return None
    try:
        client = REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL)
        account = client.get_account()
        print(f"Alpaca connected | Cash: ${float(account.cash):,.2f}")
        return client
    except Exception as e:
        print(f"Alpaca connection failed: {e}")
        return None


# Get Stocks

def get_universe():
    print("Fetching S&P 500...")
    try:
        df = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]
        df['Symbol'] = df['Symbol'].str.replace('.', '-')
        tickers = df['Symbol'].tolist()[:200]
    except:
        tickers = ['AAPL', 'MSFT', 'NVDA', 'GOOGL', 'AMZN', 'META', 'TSLA']

    print(f"Fetching metadata for {len(tickers)} stocks...")
    objs = yf.Tickers(' '.join(tickers))
    data = []
    for t in tickers:
        try:
            info = objs.tickers[t].info
            cap = info.get('marketCap', 0) or 0
            sector = info.get('sector', 'Unknown')
            if cap >= MIN_MARKET_CAP:
                data.append({'Symbol': t, 'Sector': sector, 'MarketCap': cap})
        except:
            continue
        time.sleep(0.5)
    universe = pd.DataFrame(data)
    print(f"Universe: {len(universe)} stocks")
    return universe if len(universe) > 0 else pd.DataFrame([
        {'Symbol': 'AAPL', 'Sector': 'Technology', 'MarketCap': 2.8e12},
    ])


# Indicators & Signals

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


# Execute Trades with STOP-LOSS & TAKE-PROFIT

def execute_alpaca_trades(signals_df, client):
    print(f"Submitting {len(signals_df)} trades with stop-loss...")
    cash = float(client.get_account().cash)
    total_alloc = signals_df['Allocation_%'].sum() / 100
    if total_alloc > 1.0:
        print("Scaling allocations to 100%...")
        signals_df['Allocation_%'] *= (1.0 / total_alloc)

    for _, row in signals_df.iterrows():
        ticker = row['Ticker']
        alloc_pct = row['Allocation_%'] / 100
        dollar = cash * alloc_pct
        if dollar < 10:
            continue

        try:
            # Get latest quote
            quote = client.get_latest_quote(ticker)
            ask_price = quote.ap
            if ask_price < MIN_PRICE:
                continue

            qty = int(dollar // ask_price)
            if qty == 0:
                continue

            # Calculate stop-loss and take-profit
            stop_price = round(ask_price * (1 - STOP_LOSS_PCT), 2)
            take_profit_price = round(ask_price * (1 + TAKE_PROFIT_PCT), 2) if TAKE_PROFIT_PCT > 0 else None

            # Submit bracket order
            order = client.submit_order(
                symbol=ticker,
                qty=qty,
                side='buy',
                type='market',
                time_in_force='day',
                order_class='bracket',
                stop_loss={'stop_price': stop_price},
                take_profit={'limit_price': take_profit_price} if take_profit_price else None
            )

            print(f"BUY {qty} {ticker} @ ~${ask_price:.2f}")
            print(f"  Stop-Loss: ${stop_price:.2f} | Take-Profit: ${take_profit_price:.2f}" if take_profit_price else f"  Stop-Loss: ${stop_price:.2f}")

        except APIError as e:
            print(f"Order failed {ticker}: {e}")
        except Exception as e:
            print(f"Error {ticker}: {e}")

    print("All bracket orders submitted.")


def main():
    client = get_alpaca_client() if LIVE_TRADE else None

    universe = get_universe()
    tickers = universe['Symbol'].tolist()

    print("Downloading price data...")
    data = yf.download(tickers, period="2y", progress=False, threads=False)
    if data.empty:
        print("No data.")
        return

    close = data['Close']
    volume = data['Volume']
    recent_close = close.tail(LOOKBACK_DAYS + 1)
    recent_vol = volume.tail(LOOKBACK_DAYS + 1)

    signals = []
    today = datetime.now().strftime('%Y-%m-%d')

    for ticker in tickers:
        try:
            c = recent_close[ticker].dropna()
            v = recent_vol[ticker].dropna()
            if len(c) < 100 or v[-60:].mean() < MIN_AVG_VOLUME:
                continue

            row = universe[universe['Symbol'] == ticker].iloc[0]
            price = c.iloc[-2]
            if price < MIN_PRICE:
                continue

            log_ret = np.log(c).diff().dropna()
            if len(log_ret) < 50:
                continue

            stationary, pval, decisive, _ = adf_decisive_test(log_ret)
            if not decisive:
                continue

            hist = c.iloc[:-1]
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
                    'Date': today,
                    'Ticker': ticker,
                    'Close': price,
                    'p_value': pval,
                    'Stationary': stationary,
                    'Sector': row['Sector']
                })
        except:
            continue

    if not signals:
        print("No signals today.")
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
    df['Raw_Alloc'] = df['Confidence'] / total
    sector_sum = df.groupby('Sector')['Raw_Alloc'].sum()
    for sector in sector_sum[sector_sum > MAX_SECTOR_ALLOCATION].index:
        scale = MAX_SECTOR_ALLOCATION / sector_sum[sector]
        df.loc[df['Sector'] == sector, 'Raw_Alloc'] *= scale
    df['Raw_Alloc'] = df['Raw_Alloc'] / df['Raw_Alloc'].sum()
    df['Allocation_%'] = (df['Raw_Alloc'] * 100).round(2)

    trade_file = df[['Date', 'Ticker', 'Close', 'Allocation_%', 'Sector']].sort_values('Allocation_%', ascending=False)
    trade_file.to_csv(SIGNALS_FILE, index=False)
    print(f"\nSignals saved: {SIGNALS_FILE}")
    print(trade_file.head(10).to_string(index=False))

    if LIVE_TRADE and client:
        execute_alpaca_trades(trade_file, client)


if __name__ == "__main__":
    main()