
import pickle
import pandas as pd
import fsspec
import pyarrow.parquet as pq
import pyarrow.fs
import argparse

categorical = ['PULocationID', 'DOLocationID']

def read_data(url):
    # Add Chrome-like User-Agent header
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"
    }
    with fsspec.open(url, mode='rb', headers=headers) as f:
        df = pd.read_parquet(f)

    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()
    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')

    return df

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--year', type=int, required=True)
    parser.add_argument('--month', type=int, required=True)
    args = parser.parse_args()

    year = args.year
    month = args.month

    with open('model.bin', 'rb') as f_in:
        dv, model = pickle.load(f_in)

    df = read_data(f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year}-{month:02d}.parquet')
    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = model.predict(X_val)

    print("Mean predicted duration:", round(y_pred.mean(), 2))

# import numpy as np

# print(np.std(y_pred))

# year = 2023
# month = 3

# df['ride_id'] = df.index.to_series().apply(lambda idx: f'{year:04d}/{month:02d}_{idx}')

# df_result = pd.DataFrame({
#     'ride_id': df['ride_id'],
#     'predicted_duration': y_pred
# })

# output_file = f'predictions_{year:04d}_{month:02d}.parquet'
# df_result.to_parquet(output_file, engine='pyarrow', compression=None, index=False)

# import os

# file_size_bytes = os.path.getsize(output_file)
# file_size_mb = file_size_bytes / (1024 * 1024)

# print(f"Output file size: {file_size_mb:.2f} MB")




