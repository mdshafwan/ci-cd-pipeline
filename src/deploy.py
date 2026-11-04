import shutil
import os
import datetime

source = "model.pkl"
targeted_path = "production_ready"

def deploy():
    if not os.path.exists(source):
        print("the deployment was the source model was not founded")
        exit(1)

    os.makedirs(targeted_path,exist_ok=True)

    timestamp = datetime.datetime.now().strftime(r'%Y%m%d_%H%M%S')
    version_model = f"model_{timestamp}.pkl"
    target_path = os.path.join("targeted_path",version_model)
    latest_path = os.path.join("targeted_path","latest_model.pkl")
    shutil.copy(source,target_path)
    shutil.copy(source,latest_path)

    exit(0)

if __name__ == "__main__":
    deploy()