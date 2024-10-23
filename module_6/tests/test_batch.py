import pandas as pd

from datetime import datetime
from ..app.batch import prepare_data


def dt(hour, minute, second=0):
    return datetime(2023, 1, 1, hour, minute, second)


data = [
    (None, None, dt(1, 2), dt(1, 10)),
    (1, None, dt(1, 2), dt(1, 10)),
    (1, 2, dt(2, 2), dt(2, 3)),
    (None, 1, dt(1, 2, 0), dt(1, 2, 50)),
    (2, 3, dt(1, 2, 0), dt(1, 2, 59)),
    (3, 4, dt(1, 2, 0), dt(2, 2, 1)),
]
columns=['PULocationID', 'DOLocationID', 'tpep_pickup_datetime', 'tpep_dropoff_datetime']



def test_prepare_data():
    df = pd.DataFrame(data, columns=columns)
    actual_df = prepare_data(df, ['PULocationID', 'DOLocationID'])
    expected_data = [
        {'PULocationID': '-1', 'DOLocationID': '-1', 'tpep_pickup_datetime': pd.Timestamp('2023-01-01 01:02:00'),
         'tpep_dropoff_datetime': pd.Timestamp('2023-01-01 01:10:00'), 'duration': 8.0},
        {'PULocationID': '1', 'DOLocationID': '-1', 'tpep_pickup_datetime': pd.Timestamp('2023-01-01 01:02:00'),
         'tpep_dropoff_datetime': pd.Timestamp('2023-01-01 01:10:00'), 'duration': 8.0},
        {'PULocationID': '1', 'DOLocationID': '2', 'tpep_pickup_datetime': pd.Timestamp('2023-01-01 02:02:00'),
         'tpep_dropoff_datetime': pd.Timestamp('2023-01-01 02:03:00'), 'duration': 1.0}]

    expected_columns = ['PULocationID', 'DOLocationID', 'tpep_pickup_datetime', 'tpep_dropoff_datetime', 'duration']
    expected_df = pd.DataFrame(expected_data, columns=expected_columns)

    print(len(expected_df))
    print(actual_df)

    pd.testing.assert_frame_equal(expected_df, actual_df)
