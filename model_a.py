import yfinance as yf
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
import warnings
import requests
import time

warnings.filterwarnings("ignore")


# ----------------------------------------------------------------------
# 1. Get top 100 US stocks
# ----------------------------------------------------------------------
def get_top_100_us_stocks():
    fallback = [
        'MSFT', 'AAPL', 'NVDA', 'GOOGL', 'GOOG', 'AMZN', 'META', 'BRK-B', 'AVGO', 'TSLA',
        'LLY', 'JPM', 'WMT', 'UNH', 'V', 'XOM', 'PG', 'MA', 'JNJ', 'HD',
        'COST', 'MRK', 'ABBV', 'NFLX', 'CVX', 'ORCL', 'AMD', 'BAC', 'KO', 'QCOM',
        'CRM', 'TMO', 'WFC', 'ADBE', 'ACN', 'ABT', 'DHR', 'TXN', 'NEE', 'LIN',
        'INTU', 'COP', 'C', 'GE', 'SPGI', 'PFE', 'PM', 'NOW', 'AMGN', 'DIS',
        'T', 'IBM', 'CAT', 'UPS', 'RTX', 'UNP', 'HON', 'GS', 'SBUX', 'AXP',
        'MS', 'LOW', 'PLD', 'NKE', 'SYK', 'BMY', 'MDT', 'GILD', 'MU',
        'LMT', 'ELV', 'TJX', 'BLK', 'DE', 'VRTX', 'PGR', 'REGN', 'ZTS', 'CB',
        'ADI', 'D', 'BSX', 'MMC', 'SCHW', 'HCA', 'AON', 'USB', 'CI', 'BX',
        'TGT', 'KLAC', 'MDLZ', 'BDX', 'PCAR', 'WM', 'MO', 'DUK'
    ]
    fallback = list(dict.fromkeys(fallback))[:100]

    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
        tickers = [t.replace(".", "-") for t in pd.read_html(resp.text)[0]["Symbol"].tolist()]
        print("Scraped live S&P 500 list.")
        return tickers[:200]
    except Exception as e:
        print(f"Scraping failed ({e}), using fallback.")
        return fallback


# ----------------------------------------------------------------------
# 2. ADF + Decisiveness
# ----------------------------------------------------------------------
def adf_decisive_test(series):
    series = series.dropna()
    if len(series) < 50:
        return False, np.nan, False, "Too few points"
    res = adfuller(series, autolag="AIC")
    p = res[1]
    if p <= 0.01:
        return True, p, True, "Strongly Stationary"
    elif p >= 0.10:
        return False, p, True, "Strongly Non-Stationary"
    else:
        return (p <= 0.05), p, False, f"Indecisive p={p:.4f}"


# ----------------------------------------------------------------------
# 3. Indicators
# ----------------------------------------------------------------------
def calculate_rsi(series, window=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(window).mean()
    loss = -delta.where(delta < 0, 0).rolling(window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return round(rsi.iloc[-1], 2) if len(rsi) > 0 else np.nan


def calculate_sma(series, window=20):
    return round(series.rolling(window).mean().iloc[-1], 2)


def calculate_bollinger(series, window=20, std_dev=2):
    sma = series.rolling(window).mean()
    std = series.rolling(window).std()
    lower = sma - std_dev * std
    upper = sma + std_dev * std
    return round(lower.iloc[-1], 2), round(sma.iloc[-1], 2), round(upper.iloc[-1], 2)


# ----------------------------------------------------------------------
# 4. Signals
# ----------------------------------------------------------------------
def momentum_signal(rsi, price, sma20):
    if rsi < 30 and price > sma20:
        return "BUY", "Oversold + Above SMA"
    elif rsi > 70 and price < sma20:
        return "SELL", "Overbought + Below SMA"
    elif price > sma20:
        return "HOLD", "Bullish"
    else:
        return "HOLD", "Bearish"


def mean_reversion_signal(rsi, price, lower_bb, upper_bb):
    if price < lower_bb and rsi < 30:
        return "BUY", "Oversold + Below BB"
    elif price > upper_bb and rsi > 70:
        return "SELL", "Overbought + Above BB"
    else:
        return "HOLD", "In Band"


# ----------------------------------------------------------------------
# 5. Main + Trade File
# ----------------------------------------------------------------------
def main():
    tickers = get_top_100_us_stocks()
    print(f"Analyzing {len(tickers)} stocks...\n")

    trades = []
    period = "1y"

    for i, ticker in enumerate(tickers, 1):
        try:
            data = yf.download(ticker, period=period, progress=False, auto_adjust=True)
            if data.empty or len(data) < 50:
                continue
            close = data['Close']
            price = close.iloc[-1]

            # ADF
            stationary, pval, decisive, reason = adf_decisive_test(close)
            if not decisive:
                continue  # Skip indecisive

            # Indicators
            rsi = calculate_rsi(close)
            sma20 = calculate_sma(close)
            lower_bb, _, upper_bb = calculate_bollinger(close)

            # Signal
            if not stationary:
                sig, _ = momentum_signal(rsi, price, sma20)
            else:
                sig, _ = mean_reversion_signal(rsi, price, lower_bb, upper_bb)

            if sig != "BUY":
                continue  # Long-only

            # Confidence Score
            if stationary:
                confidence = 1 / max(pval, 1e-6)  # p ≤ 0.01 → high
            else:
                confidence = pval  # p ≥ 0.10 → higher = more non-stationary

            trades.append({
                'Ticker': ticker,
                'Price': round(price, 2),
                'p_value': pval,
                'Confidence': confidence,
                'Stationary': stationary
            })

        except:
            pass

        if i % 20 == 0 or i == len(tickers):
            print(f"  Processed {i}/{len(tickers)}")
        time.sleep(0.05)

    # ------------------------------------------------------------------
    # 6. Allocate Capital Pro-Rata by Confidence
    # ------------------------------------------------------------------
    if not trades:
        print("No BUY signals with decisive ADF.")
        return

    df = pd.DataFrame(trades)
    total_confidence = df['Confidence'].sum()
    df['Allocation_%'] = (df['Confidence'] / total_confidence) * 100
    df['Allocation_%'] = df['Allocation_%'].round(2)
    df['Direction'] = 'BUY'

    # Final trade file
    trade_file = df[['Ticker', 'Direction', 'Price', 'Allocation_%']].copy()
    trade_file = trade_file.sort_values('Allocation_%', ascending=False)

    # Save CSV
    filename = "trades_long_only.csv"
    trade_file.to_csv(filename, index=False)
    print(f"\nTrade file saved: {filename}")
    print(trade_file.to_string(index=False))

    # Summary
    print(f"\nTotal BUY signals: {len(trade_file)}")
    print(f"Total capital allocated: {trade_file['Allocation_%'].sum():.2f}%")


if __name__ == "__main__":
    main()