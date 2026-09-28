from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
ARTIFACT_DIR = BASE_DIR / "artifacts"

DATASET_PATH = DATA_DIR / "PhiUSIIL_Phishing_URL_Dataset.csv"
MODEL_PATH = MODEL_DIR / "phishing_model.joblib"
METRICS_PATH = ARTIFACT_DIR / "metrics.json"
FEATURES_PATH = ARTIFACT_DIR / "selected_features.json"
SHAP_BACKGROUND_PATH = ARTIFACT_DIR / "shap_background.joblib"

RANDOM_STATE = 42
TEST_SIZE = 0.20
TOP_K_FEATURES = 15

for directory in (DATA_DIR, MODEL_DIR, ARTIFACT_DIR):
    directory.mkdir(parents=True, exist_ok=True)
