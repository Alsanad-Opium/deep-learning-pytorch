from flask import Flask
from pathlib import Path
import joblib

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "model" / "detection.pkl"

model = joblib.load(MODEL_PATH)

app = Flask(__name__)

from app.routes.predict import predict_bp

app.register_blueprint(predict_bp)