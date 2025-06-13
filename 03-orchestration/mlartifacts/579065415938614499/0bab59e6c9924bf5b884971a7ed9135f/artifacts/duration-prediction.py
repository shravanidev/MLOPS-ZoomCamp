import pickle
from pathlib import Path

import pandas as pd
import numpy as np
import xgboost as xgb

from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression


import mlflow
from prefect import flow, task

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("nyc-taxi-experiment")

models_folder = Path('models')
models_folder.mkdir(exist_ok=True)

@task
def read_dataframe(year: int, month: int) -> pd.DataFrame:
    url = f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year}-{month:02d}.parquet'
    df = pd.read_parquet(url)

    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df.duration = df.duration.dt.total_seconds() / 60
    

    df = df[(df.duration >= 1) & (df.duration <= 60)]

    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)

    # df['PU_DO'] = df['PULocationID'] + '_' + df['DOLocationID']

    return df

@task
def read_yellow_dataframe(year, month):
    url = f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year}-{month:02d}.parquet'
    df = pd.read_parquet(url)

    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df.duration = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)]

    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)

    return df


@task
def prepare_features(df: pd.DataFrame, dv: DictVectorizer = None, fit_dv: bool = True):
    categorical = ['PULocationID', 'DOLocationID']
    dicts = df[categorical].to_dict(orient='records')

    if fit_dv:
        dv = DictVectorizer()
        X = dv.fit_transform(dicts)
    else:
        X = dv.transform(dicts)

    y = df['duration'].values
    return X, y, dv


# @task
# def create_X(df, dv=None):
#     categorical = ['PU_DO']
#     numerical = ['trip_distance']
#     dicts = df[categorical + numerical].to_dict(orient='records')

#     if dv is None:
#         dv = DictVectorizer(sparse=True)
#         X = dv.fit_transform(dicts)
#     else:
#         X = dv.transform(dicts)

#     return X, dv

@task
def train_model(X_train, y_train, X_val, y_val, dv, train_size, val_size):
    mlflow.set_experiment("nyc-taxi-experiment")
    with mlflow.start_run():
        lr = LinearRegression()
        lr.fit(X_train, y_train)

        y_pred = lr.predict(X_val)
        rmse = mean_squared_error(y_val, y_pred, squared=False)

        mlflow.log_param("train_size", train_size)
        mlflow.log_param("val_size", val_size)
        mlflow.log_metric("rmse", rmse)

        mlflow.sklearn.log_model(lr, artifact_path="model")
        mlflow.log_artifact("duration-prediction.py")

        print(f"✅ Intercept of the model: {lr.intercept_}")
        print(f"✅ RMSE on validation: {rmse}")

    return lr, dv


# def pipeline(year: int, month: int):
#     df_train = read_dataframe(year, month)

#     next_year = year if month < 12 else year + 1
#     next_month = month + 1 if month < 12 else 1
#     df_val = read_dataframe(next_year, next_month)

#     X_train, dv = create_X(df_train)
#     X_val, _ = create_X(df_val, dv)

#     target = 'duration'
#     y_train = df_train[target].values
#     y_val = df_val[target].values

#     run_id = train_model(X_train, y_train, X_val, y_val, dv)
#     print(f"✅ MLflow run_id: {run_id}")
#     return run_id

@flow(name="nyc-taxi-training-pipeline")
def main(year: int, month: int):
    df = read_dataframe(year, month)
    df_train = df.sample(frac=0.8, random_state=42)
    df_val = df.drop(df_train.index)

    X_train, y_train, dv = prepare_features(df_train, fit_dv=True)
    X_val, y_val, _ = prepare_features(df_val, dv=dv, fit_dv=False)

    model, dv = train_model(X_train, y_train, X_val, y_val, dv, len(df_train), len(df_val))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Train a model to predict taxi trip duration.')
    parser.add_argument('--year', type=int, required=True, help='Year of the data to train on')
    parser.add_argument('--month', type=int, required=True, help='Month of the data to train on')
    args = parser.parse_args()

    main(args.year, args.month)

    # run_id = pipeline(year=args.year, month=args.month)
    # print(f"MLFlow run_id: {run_id}")

    # # with open("run_id.txt", "w") as f:
    # #     f.write(run_id)

    # if args.year == 2023 and args.month == 3:
    #     df_yellow = read_dataframe(args.year, args.month)
    #     df_filtered = df_yellow[(df_yellow.duration >= 1) & (df_yellow.duration <= 60)]
    #     print(f"\n🟡 Filtered record count for 2023-03 Yellow Taxi data: {len(df_filtered)}")