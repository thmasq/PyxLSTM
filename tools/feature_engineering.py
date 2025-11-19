import argparse
import warnings
from datetime import timedelta

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Feature engineering for OHLCV data")
    parser.add_argument("input_file", type=str, help="Input CSV file path")
    parser.add_argument(
        "--frequency", type=str, default="1H", help="Resampling frequency (e.g., 1H, 4H, 1D, 1W). Default: 1H"
    )
    parser.add_argument(
        "--months", type=int, default=None, help="Number of months to include (from most recent). Default: all data"
    )
    parser.add_argument("--start-date", type=str, default=None, help="Start date (YYYY-MM-DD). Overrides --months")
    parser.add_argument("--end-date", type=str, default=None, help="End date (YYYY-MM-DD). Default: most recent")
    parser.add_argument("--embedding-size", type=int, default=128, help="Size of embeddings. Default: 128")
    parser.add_argument(
        "--output-features", type=str, default="features_engineered.csv", help="Output file for engineered features"
    )
    parser.add_argument("--output-embeddings", type=str, default="embeddings.csv", help="Output file for embeddings")
    return parser.parse_args()


def load_and_prepare_data(filepath, frequency="1H", months=None, start_date=None, end_date=None):
    """Load and prepare the OHLCV data with optional filtering"""
    df = pd.read_csv(filepath)

    # Convert timestamp to datetime
    df["trading_date"] = pd.to_datetime(df["Timestamp"], unit="s")

    # Rename columns to match expected format
    df = df.rename(
        columns={
            "Open": "opening_price",
            "High": "max_price",
            "Low": "min_price",
            "Close": "last_price",
            "Volume": "total_volume",
        }
    )

    # Sort by date
    df = df.sort_values("trading_date").reset_index(drop=True)

    # Filter by date range
    if end_date:
        end_dt = pd.to_datetime(end_date)
    else:
        end_dt = df["trading_date"].max()

    if start_date:
        start_dt = pd.to_datetime(start_date)
    elif months:
        start_dt = end_dt - timedelta(days=months * 30)
    else:
        start_dt = df["trading_date"].min()

    df = df[(df["trading_date"] >= start_dt) & (df["trading_date"] <= end_dt)]

    print(f"Date range: {df['trading_date'].min()} to {df['trading_date'].max()}")
    print(f"Total records before resampling: {len(df)}")

    # Resample if needed (and frequency is not original)
    if frequency and frequency != "original":
        df = df.set_index("trading_date")

        # Resample OHLCV data
        resampled = (
            df.resample(frequency)
            .agg(
                {
                    "opening_price": "first",
                    "max_price": "max",
                    "min_price": "min",
                    "last_price": "last",
                    "total_volume": "sum",
                }
            )
            .dropna()
        )

        df = resampled.reset_index()
        print(f"Total records after resampling to {frequency}: {len(df)}")

    # Add placeholder columns for features that need them
    df["best_sell_offer"] = df["max_price"]  # Approximation
    df["best_buy_offer"] = df["min_price"]  # Approximation
    df["avg_price"] = (df["max_price"] + df["min_price"] + df["last_price"]) / 3
    df["total_quantity"] = df["total_volume"]
    df["total_trades"] = df["total_volume"] / 100  # Rough estimate
    df["trading_code"] = "ASSET"
    df["company_name"] = "Asset"

    return df


def create_price_features(df):
    """Create price-based features"""
    features = pd.DataFrame(index=df.index)

    features["price_range"] = df["max_price"] - df["min_price"]
    features["price_range_pct"] = features["price_range"] / df["opening_price"]
    features["close_open_diff"] = df["last_price"] - df["opening_price"]
    features["close_open_pct"] = features["close_open_diff"] / df["opening_price"]

    features["close_position"] = (df["last_price"] - df["min_price"]) / (df["max_price"] - df["min_price"] + 1e-10)

    features["bid_ask_spread"] = df["best_sell_offer"] - df["best_buy_offer"]
    features["bid_ask_spread_pct"] = features["bid_ask_spread"] / df["last_price"]

    return features


def create_returns_features(df, windows=[1, 2, 3, 5, 10, 20, 30, 60, 90]):
    """Create return-based features"""
    features = pd.DataFrame(index=df.index)

    for window in windows:
        features[f"return_{window}d"] = df["last_price"].pct_change(window)
        features[f"log_return_{window}d"] = np.log(df["last_price"] / df["last_price"].shift(window))
        features[f"volatility_{window}d"] = df["last_price"].pct_change().rolling(window).std()
        features[f"skew_{window}d"] = df["last_price"].pct_change().rolling(window).skew()
        features[f"kurt_{window}d"] = df["last_price"].pct_change().rolling(window).kurt()

    return features


def create_moving_average_features(df, windows=[5, 10, 20, 50, 200]):
    """Create moving average features"""
    features = pd.DataFrame(index=df.index)

    for window in windows:
        ma = df["last_price"].rolling(window).mean()
        features[f"ma_{window}"] = ma
        features[f"price_to_ma_{window}"] = df["last_price"] / ma - 1

        ema = df["last_price"].ewm(span=window, adjust=False).mean()
        features[f"ema_{window}"] = ema
        features[f"price_to_ema_{window}"] = df["last_price"] / ema - 1

    return features


def create_volume_features(df, windows=[5, 10, 20]):
    """Create volume-based features"""
    features = pd.DataFrame(index=df.index)

    features["volume"] = df["total_volume"]
    features["quantity"] = df["total_quantity"]
    features["avg_trade_size"] = df["total_volume"] / (df["total_trades"] + 1)

    for window in windows:
        features[f"volume_ma_{window}"] = df["total_volume"].rolling(window).mean()
        features[f"volume_ratio_{window}"] = df["total_volume"] / (features[f"volume_ma_{window}"] + 1)
        features[f"vwap_{window}"] = (df["avg_price"] * df["total_volume"]).rolling(window).sum() / df[
            "total_volume"
        ].rolling(window).sum()

    features["price_volume_trend"] = df["last_price"].pct_change() * df["total_volume"]

    return features


def create_technical_indicators(df):
    """Create technical indicator features"""
    features = pd.DataFrame(index=df.index)

    # RSI
    for period in [7, 14, 21]:
        delta = df["last_price"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / (loss + 1e-10)
        features[f"rsi_{period}"] = 100 - (100 / (1 + rs))

    # MACD
    for fast, slow in [(12, 26), (5, 35)]:
        ema_fast = df["last_price"].ewm(span=fast, adjust=False).mean()
        ema_slow = df["last_price"].ewm(span=slow, adjust=False).mean()
        features[f"macd_{fast}_{slow}"] = ema_fast - ema_slow
        features[f"macd_signal_{fast}_{slow}"] = features[f"macd_{fast}_{slow}"].ewm(span=9, adjust=False).mean()
        features[f"macd_diff_{fast}_{slow}"] = features[f"macd_{fast}_{slow}"] - features[f"macd_signal_{fast}_{slow}"]

    # Bollinger Bands
    ma_20 = df["last_price"].rolling(20).mean()
    std_20 = df["last_price"].rolling(20).std()
    features["bb_upper"] = ma_20 + (std_20 * 2)
    features["bb_lower"] = ma_20 - (std_20 * 2)
    features["bb_width"] = (features["bb_upper"] - features["bb_lower"]) / ma_20
    features["bb_position"] = (df["last_price"] - features["bb_lower"]) / (
        features["bb_upper"] - features["bb_lower"] + 1e-10
    )

    # Stochastic Oscillator
    low_14 = df["min_price"].rolling(14).min()
    high_14 = df["max_price"].rolling(14).max()
    features["stoch_k"] = 100 * (df["last_price"] - low_14) / (high_14 - low_14 + 1e-10)
    features["stoch_d"] = features["stoch_k"].rolling(3).mean()

    # ATR
    high_low = df["max_price"] - df["min_price"]
    high_close = np.abs(df["max_price"] - df["last_price"].shift())
    low_close = np.abs(df["min_price"] - df["last_price"].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    features["atr_14"] = true_range.rolling(14).mean()
    features["atr_20"] = true_range.rolling(20).mean()

    # Momentum indicators
    for period in [10, 20, 30]:
        features[f"momentum_{period}"] = df["last_price"] - df["last_price"].shift(period)
        features[f"roc_{period}"] = (
            (df["last_price"] - df["last_price"].shift(period)) / df["last_price"].shift(period) * 100
        )

    # Williams %R
    for period in [14, 20]:
        highest = df["max_price"].rolling(period).max()
        lowest = df["min_price"].rolling(period).min()
        features[f"williams_r_{period}"] = -100 * (highest - df["last_price"]) / (highest - lowest + 1e-10)

    return features


def create_temporal_features(df):
    """Create time-based features"""
    features = pd.DataFrame(index=df.index)

    features["day_of_week"] = df["trading_date"].dt.dayofweek
    features["day_of_month"] = df["trading_date"].dt.day
    features["month"] = df["trading_date"].dt.month
    features["quarter"] = df["trading_date"].dt.quarter
    features["year"] = df["trading_date"].dt.year
    features["week_of_year"] = df["trading_date"].dt.isocalendar().week
    features["hour"] = df["trading_date"].dt.hour

    features["day_sin"] = np.sin(2 * np.pi * features["day_of_week"] / 7)
    features["day_cos"] = np.cos(2 * np.pi * features["day_of_week"] / 7)
    features["month_sin"] = np.sin(2 * np.pi * features["month"] / 12)
    features["month_cos"] = np.cos(2 * np.pi * features["month"] / 12)
    features["hour_sin"] = np.sin(2 * np.pi * features["hour"] / 24)
    features["hour_cos"] = np.cos(2 * np.pi * features["hour"] / 24)

    return features


def create_lag_features(df, lags=[1, 2, 3, 5, 10]):
    """Create lagged price features"""
    features = pd.DataFrame(index=df.index)

    for lag in lags:
        features[f"price_lag_{lag}"] = df["last_price"].shift(lag)
        features[f"volume_lag_{lag}"] = df["total_volume"].shift(lag)
        features[f"trades_lag_{lag}"] = df["total_trades"].shift(lag)
        features[f"high_lag_{lag}"] = df["max_price"].shift(lag)
        features[f"low_lag_{lag}"] = df["min_price"].shift(lag)

    return features


def create_all_features(df, output_path):
    """Main function to create all features"""
    print("\nCreating features...")
    feature_sets = []

    feature_sets.append(create_price_features(df))
    feature_sets.append(create_returns_features(df))
    feature_sets.append(create_moving_average_features(df))
    feature_sets.append(create_volume_features(df))
    feature_sets.append(create_technical_indicators(df))
    feature_sets.append(create_temporal_features(df))
    feature_sets.append(create_lag_features(df))

    all_features = pd.concat(feature_sets, axis=1)

    final_df = pd.concat(
        [
            df[["trading_date", "trading_code", "company_name", "last_price", "total_volume"]],
            all_features,
        ],
        axis=1,
    )

    print(f"Total features created: {len(all_features.columns)}")

    final_df = final_df.fillna(method="bfill").fillna(0)
    final_df.to_csv(output_path, index=False)
    print(f"Features saved to {output_path}")

    return final_df


def create_embeddings(features_df, embedding_size=128, output_path="embeddings.csv"):
    """Create fixed-size embeddings using PCA without look-ahead bias"""
    print("\nCreating embeddings (no look-ahead bias)...")

    numeric_cols = features_df.select_dtypes(include=[np.number]).columns.tolist()
    exclude_cols = ["year", "day_of_month", "week_of_year"]
    numeric_cols = [col for col in numeric_cols if col not in exclude_cols]

    X = features_df[numeric_cols].fillna(0).values
    n_samples, n_features = X.shape

    print(f"Number of features: {n_features}")
    print(f"Total samples: {n_samples}")

    min_train_size = min(252, max(60, n_samples // 10))
    print(f"Minimum training size: {min_train_size} samples")

    embeddings = np.zeros((n_samples, embedding_size))
    use_pca = n_features >= embedding_size

    if not use_pca:
        print(f"WARNING: Only {n_features} features, using zero-padding")

    print("\nProcessing with expanding window...")

    for i in range(n_samples):
        if i < min_train_size:
            X_train = X[: i + 1]
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_train)

            if use_pca and i >= 10:
                n_components = min(embedding_size, X_scaled.shape[0] - 1, X_scaled.shape[1])
                if n_components > 0:
                    pca = PCA(n_components=n_components)
                    if X_scaled.shape[0] > 1:
                        pca.fit(X_scaled[:-1])
                        emb = pca.transform(X_scaled[-1:])
                        embeddings[i, : emb.shape[1]] = emb[0]
                    else:
                        embeddings[i, : min(n_features, embedding_size)] = X_scaled[
                            -1, : min(n_features, embedding_size)
                        ]
                else:
                    embeddings[i, : min(n_features, embedding_size)] = X_scaled[-1, : min(n_features, embedding_size)]
            else:
                embeddings[i, : min(n_features, embedding_size)] = X_scaled[-1, : min(n_features, embedding_size)]
        else:
            X_past = X[:i]
            X_current = X[i : i + 1]

            scaler = StandardScaler()
            X_past_scaled = scaler.fit_transform(X_past)
            X_current_scaled = scaler.transform(X_current)

            if use_pca:
                pca = PCA(n_components=embedding_size)
                pca.fit(X_past_scaled)
                embedding_current = pca.transform(X_current_scaled)
                embeddings[i] = embedding_current[0]
            else:
                embeddings[i, :n_features] = X_current_scaled[0]

        if (i + 1) % 250 == 0 or i == n_samples - 1:
            print(f"  Processed {i + 1}/{n_samples} samples ({100 * (i + 1) / n_samples:.1f}%)")

    print(f"✓ Final embeddings shape: {embeddings.shape}")

    embedding_cols = [f"emb_{i}" for i in range(embedding_size)]
    embeddings_df = pd.DataFrame(embeddings, columns=embedding_cols)

    embeddings_df = pd.concat(
        [
            features_df[["trading_date", "trading_code", "company_name", "last_price"]].reset_index(drop=True),
            embeddings_df,
        ],
        axis=1,
    )

    embeddings_df.to_csv(output_path, index=False)
    print(f"✓ Embeddings saved to {output_path}")

    return embeddings_df, None, None


def main():
    args = parse_arguments()

    print(f"Loading data from {args.input_file}...")
    print(f"Frequency: {args.frequency}")
    if args.months:
        print(f"Date range: Last {args.months} months")
    elif args.start_date:
        print(f"Date range: {args.start_date} to {args.end_date or 'latest'}")

    df = load_and_prepare_data(
        args.input_file,
        frequency=args.frequency,
        months=args.months,
        start_date=args.start_date,
        end_date=args.end_date,
    )

    features_df = create_all_features(df, args.output_features)

    embeddings_df, _, _ = create_embeddings(
        features_df, embedding_size=args.embedding_size, output_path=args.output_embeddings
    )

    print("\n✓ Processing complete!")
    print(f"  Features: {args.output_features}")
    print(f"  Embeddings: {args.output_embeddings}")


if __name__ == "__main__":
    main()
