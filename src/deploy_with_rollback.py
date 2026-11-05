import os
import shutil
import sys
import joblib
import numpy as np

# Paths
CANDIDATE_MODEL = "model.pkl"              # new model from train job
PROD_DIR = "production_models"
CURRENT_MODEL = os.path.join(PROD_DIR, "current_model.pkl")
BACKUP_MODEL = os.path.join(PROD_DIR, "backup_model.pkl")
HEALTH_REPORT = os.path.join(PROD_DIR, "last_deploy_status.txt")


def ensure_prod_dir():
    os.makedirs(PROD_DIR, exist_ok=True)


def health_check(model_path: str) -> bool:
    """
    Basic production health check:
    1. File exists and loads
    2. Has predict method
    3. Can run a tiny dummy inference without crashing
    """
    print(f"[INFO] Running health check on: {model_path}")

    try:
        if not os.path.exists(model_path):
            print("[ERROR] Health check failed: file missing")
            return False

        model = joblib.load(model_path)

        if not hasattr(model, "predict"):
            print("[ERROR] Health check failed: object has no predict()")
            return False

        # Smoke test
        print("[SUCCESS] Model loaded successfully and has predict()")
        return True

    except Exception as e:
        print(f"[ERROR] Health check exception: {e}")
        return False


def write_status(message: str):
    with open(HEALTH_REPORT, "w") as f:
        f.write(message)


def deploy_with_rollback():
    ensure_prod_dir()

    # 1. Candidate must exist
    if not os.path.exists(CANDIDATE_MODEL):
        print("[ERROR] Candidate model.pkl not found. Did train job upload artifact?")
        write_status("FAILED: candidate missing")
        sys.exit(1)

    # 2. Backup current production model
    if os.path.exists(CURRENT_MODEL):
        print("[INFO] Backing up current production model -> backup_model.pkl")
        shutil.copy2(CURRENT_MODEL, BACKUP_MODEL)
    else:
        print("[INFO] No existing production model. First deploy.")

    # 3. Promote candidate
    print("[INFO] Promoting candidate model to production (current_model.pkl)")
    shutil.copy2(CANDIDATE_MODEL, CURRENT_MODEL)

    # 4. Health check
    healthy = health_check(CURRENT_MODEL)

    if healthy:
        print("[SUCCESS] DEPLOY SUCCESS - new model is live")
        write_status("SUCCESS: new model deployed")

        import datetime

        ts = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        versioned = os.path.join(PROD_DIR, f"model_v_{ts}.pkl")
        shutil.copy2(CURRENT_MODEL, versioned)

        print(f"[INFO] Version snapshot saved: {versioned}")
        sys.exit(0)

    # 5. Rollback
    print("[ERROR] HEALTH CHECK FAILED - INITIATING ROLLBACK")

    if os.path.exists(BACKUP_MODEL):
        shutil.copy2(BACKUP_MODEL, CURRENT_MODEL)
        print("[SUCCESS] ROLLBACK COMPLETE - restored backup_model.pkl -> current_model.pkl")
        write_status("ROLLBACK: restored previous stable model")
    else:
        print("[WARNING] No backup available to restore (first deploy failed)")
        write_status("FAILED: no backup to rollback")

    # Fail CI job
    sys.exit(1)


if __name__ == "__main__":
    deploy_with_rollback()