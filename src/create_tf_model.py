import os
import tensorflow as tf
from tensorflow.keras import layers 
from tensorflow.keras import Sequential
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.compose import ColumnTransformer
import pandas as pd
import joblib

df = pd.read_csv('../data/job_salary_prediction_dataset.csv')
df.head()

x = df.drop('salary',axis=1)
y = df['salary']

num = x.select_dtypes(include=['int64','float64']).columns.tolist()
cat = x.select_dtypes(include=['object']).columns.tolist()

num_pipe = Pipeline([('impute',SimpleImputer(strategy='median')),('scale',StandardScaler())])
cat_pipe = Pipeline([('impute',SimpleImputer(strategy='most_frequent')),('one',OneHotEncoder(handle_unknown='ignore',drop='first'))])

preprocess = ColumnTransformer([('num',num_pipe,num),('cat',cat_pipe,cat)])
x_preprocess = preprocess.fit_transform(x)

model = Sequential([
layers.Dense(128, activation='relu'),
layers.Dense(64, activation='relu'),
layers.Dense(32, activation='relu'),
layers.Dense(1)
])

model.compile(optimizer='adam',loss='mse',metrics=[tf.keras.metrics.R2Score(name='r2')])
model.fit(x_preprocess,y,epochs=3)

model_dir = 'saved_models/my_models/1'
os.makedirs(model_dir,exist_ok=True)
joblib.dump(preprocess, "preprocess.pkl")
model.export(model_dir)