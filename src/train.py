import os
import joblib
import mlflow.sklearn
import optuna
import mlflow
from sklearn.metrics import f1_score,recall_score,accuracy_score 
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,OneHotEncoder,LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import pandas as pd
from mlflow.models import infer_signature
import pytest

def train():
    data_path =  "data/customer_churn_dataset-testing-master.csv"
    
    if not os.path.exists(data_path):
        print("the empty dataset")
        exit(1)

    df = pd.read_csv(data_path)

    print(df.head())
    print(df["Churn"].value_counts())

    x = df.drop(["Churn","CustomerID"],axis=1)

    le = LabelEncoder()
    y = le.fit_transform(df["Churn"])

    mlflow.set_tracking_uri('file:./mlruns')
    mlflow.set_experiment('churn_dataset')


    num = x.select_dtypes(include=['int64','float64']).columns.tolist()
    cat = x.select_dtypes(include=['object']).columns.tolist()
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42,stratify=y)

    num_pipe = Pipeline([("impute",SimpleImputer(strategy="median")),("scale",StandardScaler())])

    cat_pipe = Pipeline([("impute",SimpleImputer(strategy="most_frequent")),("one",OneHotEncoder(handle_unknown="ignore",drop="first"))])

    preprocess = ColumnTransformer([("num",num_pipe,num),("cat",cat_pipe,cat)])
    def objective(trials):
        with mlflow.start_run(nested=True):
            params = {
                "learning_rate":trials.suggest_float('learning_rate',0.1,0.5),
                "n_estimators":trials.suggest_int('n_estimators',100,300),
                "max_depth":trials.suggest_int("max_depth",6,8),
            }

            model = Pipeline([("preprocess",preprocess),("model",XGBClassifier(**params))])
            mlflow.log_params(params)
            model.fit(x_train,y_train)
            pred = model.predict(x_test)
            acc = accuracy_score(y_test,pred)
            f1 = f1_score(y_test,pred)
            recall = recall_score(y_test,pred)
            mlflow.log_metric('accuracy',acc)
            mlflow.log_metric('f1',f1)  
            mlflow.log_metric('recall',recall)
            mlflow.log_param('dataset',"churn")
            
            return f1

    with mlflow.start_run(run_name="optuna_search"):
        study = optuna.create_study(direction="maximize")
        study.optimize(objective,n_trials=10)


        best_model = Pipeline([('preprocess',preprocess),('model',XGBClassifier(**study.best_params))])
        best_model.fit(x_train,y_train)
        pred = best_model.predict(x_test)
        best_accuracy = accuracy_score(y_test,pred)
        best_f1 = f1_score(y_test,pred)
        best_recall = recall_score(y_test,pred)

        mlflow.log_params(study.best_params)
        mlflow.log_metric('accuracy',best_accuracy)
        mlflow.log_metric('f1',best_f1)
        mlflow.log_metric("recall",best_recall)
        signature = infer_signature(x_train,best_model.predict(x_train))
        mlflow.sklearn.log_model(best_model,name='model',serialization_format='pickle',signature=signature)
        joblib.dump(best_model, "model.pkl")
        joblib.dump((x_test, y_test), "set.pkl")

if __name__ == "__main__":
    train()