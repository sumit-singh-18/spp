import pandas as pd
import numpy as np
import ta
from transformers import pipeline

# Load data
df_ohlcv = pd.read_csv("Project/ohlcv_data.csv")
dn = pd.read_csv("Project/news_data.csv")

# Sentiment analysis
finbert = pipeline("sentiment-analysis", model="yiyanghkust/finbert-tone")

sentiments = []
sentiment_scores = []

for h in dn['Title']:
    result = finbert(h)[0]
    sentiments.append(result['label'])
    sentiment_scores.append(result['score'])

dn['sentiment'] = sentiments
dn['sentiment_score'] = sentiment_scores

# Aggregate sentiment per date
agg_sentiment = dn.groupby('Date').agg(
    avg_sentiment_score=('sentiment_score', 'mean'),
    count_news=('Title', 'count'),
    mode_sentiment=('sentiment', lambda x: x.mode()[0] if not x.mode().empty else 'Neutral')
).reset_index()

# Merge with OHLCV
dm = pd.merge(agg_sentiment, df_ohlcv, how="right", on="Date")
dm = dm.fillna({'avg_sentiment_score': 0, 'count_news': 0, 'mode_sentiment': 'Neutral'})

# Quantitative features
for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
    dm[col] = pd.to_numeric(dm[col], errors='coerce')

dm['Return'] = dm['Close'].pct_change()
dm['Target'] = np.where(dm['Return'].shift(-1) > 0, 1, 0)

for lag in range(1, 6):
    dm[f'Return_lag{lag}'] = dm['Return'].shift(lag)
    dm[f'Volume_lag{lag}'] = dm['Volume'].shift(lag)

# Technical indicators
dm['SMA_10'] = ta.trend.sma_indicator(dm['Close'], window=10)
dm['EMA_10'] = ta.trend.ema_indicator(dm['Close'], window=10)
dm['RSI_14'] = ta.momentum.rsi(dm['Close'], window=14)
dm['MACD'] = ta.trend.macd(dm['Close'])
dm['MACD_signal'] = ta.trend.macd_signal(dm['Close'])

bb = ta.volatility.BollingerBands(dm['Close'], window=20, window_dev=2)
dm['BB_High'] = bb.bollinger_hband()
dm['BB_Low'] = bb.bollinger_lband()
dm['ATR_14'] = ta.volatility.average_true_range(dm['High'], dm['Low'], dm['Close'], window=14)

dm = dm.dropna().reset_index(drop=True)

# Save final dataset
dm.to_csv('final_dataset.csv', index=False)

# Basic analysis
print("Head of final dataset:")
print(dm.head())
print("\nCorrelation between avg_sentiment_score and Return:")
print(dm[['avg_sentiment_score', 'Return']].corr())
print("\nMean sentiment score by target:")
print(dm.groupby('Target')['avg_sentiment_score'].mean())
