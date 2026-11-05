import os
import shutil
import joblib

PROD_DIR = "production_models"
NEW_MODEL = "model.pkl"
PROD_MODEL = os.path.join(PROD_DIR, "current_model.pkl")
BACKUP_MODEL = os.path.join(PROD_DIR, "backup_model.pkl")

# Ensure production folder exists
os.makedirs(PROD_DIR, exist_ok=True)

# 1. Backup existing production model if it exists
if os.path.exists(PROD_MODEL):
    print(" Backing up current production model...")
    shutil.copy(PROD_MODEL, BACKUP_MODEL)

try:
    # 2. Try loading the newly trained model (Health Check)
    print(" Testing new candidate model...")
    model = joblib.load(NEW_MODEL)
    
    # 3. If valid, promote it to current_model.pkl
    shutil.copy(NEW_MODEL, PROD_MODEL)
    print(" DEPLOY SUCCESS: New model is live in production!")
    exit(0)

except Exception as e:
    print(f" DEPLOY FAILED: {e}")
    print(" INITIATING ROLLBACK...")
    
    # 4. Rollback: Restore the backup model if available
    if os.path.exists(BACKUP_MODEL):
        shutil.copy(BACKUP_MODEL, PROD_MODEL)
        print(" ROLLBACK COMPLETE: Restored previous stable model.")
    else:
        print(" No backup model available to restore.")
        
    exit(1) 