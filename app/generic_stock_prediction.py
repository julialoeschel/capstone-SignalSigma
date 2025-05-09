import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt

# Directory containing local CSV files for each ticker
data_dir = '../data/stock'

def load_stock_data(ticker, start=None, end=None):
    """
    Load historical stock data for a given ticker from a local CSV file.
    Handles files with extra header rows by skipping metadata lines.
    Args:
        ticker (str): Stock symbol, e.g., 'AAPL'.
        start (str): Start date 'YYYY-MM-DD', optional.
        end (str): End date 'YYYY-MM-DD', optional.
    Returns:
        pd.DataFrame: Indexed by Date, columns ['Open','High','Low','Close','Volume'].
    """
    file_path = os.path.join(data_dir, f"{ticker}_stock.csv")
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Data file for ticker '{ticker}' not found at {file_path}")
    # Read CSV, skipping metadata rows
    df = pd.read_csv(
        file_path,
        skiprows=[1, 2],      # skip metadata rows
        header=0,              # use first line as header
        parse_dates=[0],       # parse first column as dates
        index_col=0            # set first column as index
    )
    # Rename index and clean up
    df.index.name = 'Date'
    df.sort_index(inplace=True)
    # Keep only relevant columns
    expected_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    missing = [col for col in expected_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in data file: {missing}")
    df = df[expected_cols]
    # Filter by date range
    if start:
        df = df[df.index >= pd.to_datetime(start)]
    if end:
        df = df[df.index <= pd.to_datetime(end)]
    if df.empty:
        raise ValueError(f"No data for ticker '{ticker}' in the specified date range.")
    return df

def prepare_data(df, feature_col='Close', seq_len=60):
    data = df[[feature_col]].values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(data)
    X, y = [], []
    for i in range(seq_len, len(scaled)):
        X.append(scaled[i-seq_len:i, 0])
        y.append(scaled[i, 0])
    X, y = np.array(X), np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    return X, y, scaler


def build_model(input_shape):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        LSTM(50),
        Dense(25, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


def predict_future(model, last_sequence, scaler, n_days=1):
    preds = []
    seq = last_sequence.copy()
    for _ in range(n_days):
        X_in = seq.reshape((1, seq.shape[0], 1))
        yhat = model.predict(X_in, verbose=0)[0, 0]
        preds.append(yhat)
        seq = np.append(seq[1:], yhat)
    preds = np.array(preds).reshape(-1, 1)
    return scaler.inverse_transform(preds).flatten().tolist()


def train_and_predict(
    ticker,
    start=None,
    end=None,
    seq_len=60,
    epochs=10,
    batch_size=32,
    predict_days=1
):
    df = load_stock_data(ticker, start, end)
    X, y, scaler = prepare_data(df, seq_len=seq_len)
    model = build_model((X.shape[1], 1))
    model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=1)
    last_seq = X[-1].flatten()
    # Generate next business days
    next_start = df.index[-1] + pd.Timedelta(days=1)
    future_dates = pd.bdate_range(start=next_start, periods=predict_days)
    future_preds = predict_future(model, last_seq, scaler, n_days=predict_days)
    result = pd.DataFrame({'Date': future_dates, 'Predicted_Close': future_preds})
    result.set_index('Date', inplace=True)
    return result


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Generic Stock Price Prediction')
    parser.add_argument('--ticker', type=str, required=True, help='Stock ticker')
    parser.add_argument('--start', type=str, default=None, help='Start date YYYY-MM-DD')
    parser.add_argument('--end', type=str, default=None, help='End date YYYY-MM-DD')
    parser.add_argument('--seq_len', type=int, default=60, help='Sequence length')
    parser.add_argument('--epochs', type=int, default=10, help='Training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--predict_days', type=int, default=1, help='Days ahead to predict')
    args = parser.parse_args()
    predictions = train_and_predict(
        args.ticker,
        start=args.start,
        end=args.end,
        seq_len=args.seq_len,
        epochs=args.epochs,
        batch_size=args.batch_size,
        predict_days=args.predict_days
    )
    print(predictions)
