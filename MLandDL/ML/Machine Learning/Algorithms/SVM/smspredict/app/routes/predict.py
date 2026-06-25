from flask import request,jsonify,Blueprint
from app import model
predict_bp = Blueprint('predict',__name__,url_prefix= '/predict')



@predict_bp.route('/', methods = ['POST'], strict_slashes = False)
def predict():
    data  = request.get_json()
    
    text = data.get('text')
    if not text:
        return jsonify({'message':"Invalid Input"}),400
    
    
    result = model.predict([text])[0]
    
    return jsonify({"message":f"The input {data} is {result}"})
    