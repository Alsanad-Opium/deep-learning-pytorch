from flask import request,jsonify,Blueprint

predict_bp = Blueprint('predict',__name__,url_prefix= '/predict')


@predict_bp.route('/', method = ['POST'], strict_slashes = False)
def predict():
    