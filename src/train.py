from sklearn.model_selection import train_test_split
import pandas as pd
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import r2_score
import joblib

df = pd.read_csv('data/job_salary_prediction_dataset.csv')
df.head()

x = df.drop('salary',axis=1)
y = df['salary']

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

num = x.select_dtypes(include=['int64','float64']).columns.tolist()
cat = x.select_dtypes(include=['object']).columns.tolist()

print(num)
print(cat)

num_pipe = Pipeline([('impute',SimpleImputer(strategy='median')),('scale',StandardScaler())])
cat_pipe = Pipeline([('impute',SimpleImputer(strategy='most_frequent')),('one',OneHotEncoder(handle_unknown='ignore',drop='first'))])

preprocess = ColumnTransformer([('num',num_pipe,num),('cat',cat_pipe,cat)])

model = Pipeline([('preprocess',preprocess),('model',XGBRegressor(learning_rate=0.1,max_depth=6,n_estimators=200))])

model.fit(x_train,y_train)
joblib.dump(model,'model.pkl')
joblib.dump((x_test,y_test),'set.pkl')

print("model_trained and saved inclduing the testdataset")