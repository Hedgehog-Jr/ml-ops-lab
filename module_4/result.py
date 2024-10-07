import pickle
import pandas as pd
import sys
import os
import logging


logger = logging.getLogger('my_application')
logger.info('Start execution')


MODEL_FILE = os.getenv('MODEL_FILE', 'model.bin')
categorical = ['PULocationID', 'DOLocationID']
year = int(sys.argv[1])
month = int(sys.argv[2])
input_path = f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet'
output_path = f'output/yellow_tripdata_{year:04d}-{month:02d}.parquet'


def read_data(filename):
    df = pd.read_parquet(filename)

    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')

    return df


with open('model.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)

df = read_data(input_path)

dicts = df[categorical].to_dict(orient='records')
X_val = dv.transform(dicts)
y_pred = model.predict(X_val)

df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')
df_output = pd.DataFrame()
df_output['ride_id'] = df['ride_id']
df_output['predicted_duration'] = y_pred

df_output.to_parquet(
    output_path,
    engine='pyarrow',
    compression=None,
    index=False
)

print('Predicted mean duration:', y_pred.mean())
