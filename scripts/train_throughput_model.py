import os
import glob
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import joblib


ROOT_DIR   = os.path.dirname(os.path.dirname(__file__))
DATA_DIR   = os.path.join(ROOT_DIR, "data", "throughput")
ONTIME_DIR = os.path.join(ROOT_DIR, "data", "ontime")
MODELS_DIR = os.path.join(ROOT_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)


def aggregate_airport(path):
    df = pd.read_csv(path)
    # Expect columns: Date, Hour, then many checkpoint columns
    if 'Date' not in df.columns or 'Hour' not in df.columns:
        raise ValueError(f"Unexpected columns in {path}: {df.columns.tolist()}")
    id_vars = ['Date', 'Hour']
    value_cols = [c for c in df.columns if c not in id_vars]
    # Melt to long then sum across checkpoints for each Date+Hour
    long = df.melt(id_vars=id_vars, value_vars=value_cols, var_name='checkpoint', value_name='value')
    long['value'] = pd.to_numeric(long['value'], errors='coerce').fillna(0.0)
    agg = long.groupby(['Date', 'Hour'], as_index=False)['value'].sum()
    # create a datetime index
    agg['datetime'] = pd.to_datetime(agg['Date'].astype(str) + ' ' + agg['Hour'].astype(str), errors='coerce')
    agg = agg.dropna(subset=['datetime']).sort_values('datetime')
    agg = agg[['datetime', 'value']].rename(columns={'value': 'throughput'})
    return agg


def make_features(df):
    df = df.copy()
    df['hour'] = df['datetime'].dt.hour
    df['dow'] = df['datetime'].dt.dayofweek
    df['month'] = df['datetime'].dt.month
    df['lag_1'] = df['throughput'].shift(1)
    df['lag_24'] = df['throughput'].shift(24)
    df['rmean_24'] = df['throughput'].rolling(24, min_periods=1).mean().shift(1)
    df = df.dropna().reset_index(drop=True)
    X = df[['hour', 'dow', 'month', 'lag_1', 'lag_24', 'rmean_24']]
    y = df['throughput']
    return X, y


def train_and_evaluate(df, airport_code):
    X, y = make_features(df)
    # time-based split: keep last 14 days (approx) as test if enough data
    if len(X) < 200:
        test_size = 0.25
    else:
        test_size = 0.2
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)

    # Baselines
    # naive: previous hour (lag_1)
    naive_pred = X_test['lag_1'].values
    naive_mae = mean_absolute_error(y_test, naive_pred)
    mean_pred = np.repeat(y_train.mean(), len(y_test))
    mean_mae = mean_absolute_error(y_test, mean_pred)

    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)

    out = {
        'airport': airport_code,
        'mae_model': float(mae),
        'mae_naive': float(naive_mae),
        'mae_mean': float(mean_mae),
        'train_rows': int(len(X_train)),
        'test_rows': int(len(X_test)),
    }

    model_path = os.path.join(MODELS_DIR, f"throughput_{airport_code}.pkl")
    joblib.dump(model, model_path)
    return out


def main(run_all=True, airport=None):
    files = sorted(glob.glob(os.path.join(DATA_DIR, 'TsaThroughput.*.csv')))
    results = []
    for p in files:
        code = os.path.basename(p).split('.')[-2]
        if (not run_all) and airport and code != airport:
            continue
        print(f"Processing {code} from {p}")
        try:
            agg = aggregate_airport(p)
            if agg.empty:
                print(f"No rows after aggregation for {code}, skipping")
                continue
            res = train_and_evaluate(agg, code)
            print(f"{code}: model MAE={res['mae_model']:.2f}, naive MAE={res['mae_naive']:.2f}, mean MAE={res['mae_mean']:.2f}")
            results.append(res)
        except Exception as e:
            print(f"Error processing {code}: {e}")

    if results:
        dfres = pd.DataFrame(results).sort_values('mae_model')
        print('\nSummary (models vs naive):')
        print(dfres[['airport', 'mae_model', 'mae_naive', 'mae_mean', 'train_rows', 'test_rows']].to_string(index=False))
    else:
        print('No models trained.')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--airport', help='Airport code to run (e.g., LAX)')
    args = parser.parse_args()
    if args.airport:
        main(run_all=False, airport=args.airport)
    else:
        main()
