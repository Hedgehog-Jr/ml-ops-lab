#!/usr/bin/env python
# coding: utf-8

# In[7]:


#get_ipython().system('pip freeze | grep scikit-learn')
# !pip install scikit-learn==1.5.0


# In[2]:


#get_ipython().system('python -V')


# In[3]:


import pickle
import pandas as pd
import sys
import os


# In[8]:

MODEL_FILE = os.getenv('MODEL_FILE', 'model.bin')
with open('model.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)


# In[9]:


categorical = ['PULocationID', 'DOLocationID']

def read_data(filename):
    df = pd.read_parquet(filename)
    
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df


# In[26]:


year = int(sys.argv[1])
month = int(sys.argv[2])
input_path = f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet'
output_path = f'output/yellow_tripdata_{year:04d}-{month:02d}.parquet'


# In[27]:


df = read_data(input_path)


# In[17]:


dicts = df[categorical].to_dict(orient='records')
X_val = dv.transform(dicts)
y_pred = model.predict(X_val)


# <h3>Q1. Notebook. The standard deviation of the predicted duration for this dataset is</h3>

# In[18]:


y_pred.std()


# <h3>Q2. Preparing the output</h3>

# In[29]:


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

#get_ipython().system('ls -lh output')

print('Predicted mean duration:', y_pred.mean())

# In[1]:


# !jupyter nbconvert --to script starter.ipynb

