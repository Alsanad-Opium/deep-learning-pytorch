from flask import Flask
from pathlib import Path
import joblib

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "model" / "detection.pkl"

model = joblib.load(MODEL_PATH)

FRONTEND_DIR = BASE_DIR.parent.parent / "frontend"
app = Flask(
    __name__,
    static_folder=str(FRONTEND_DIR),
    static_url_path="/static",
)

@app.route("/", methods=["GET"])
def index():
    return app.send_static_file("index.html")

from app.routes.predict import predict_bp

app.register_blueprint(predict_bp)