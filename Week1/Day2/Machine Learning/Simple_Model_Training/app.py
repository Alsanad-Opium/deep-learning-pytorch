from flask import Flask,request,jsonify, render_template
import numpy as np 
import pickle

app = Flask(__name__)

model = pickle.load(open("placement_model.pkl","rb"))
scalar = pickle.load(open("scalar.pkl", "rb"))



@app.route('/')

def home():
    return render_template('index.html')


@app.route('/predict',methods = ['POST'])
def predict():
    
    cgpa = float(request.form['cgpa'])
    iq = float(request.form['iq'])
    
    input_data = np.array([[cgpa,iq]])
    scaled_data = scalar.transform(input_data)
    
    prediction = model.predict(scaled_data)[0]
    
    if prediction == 1:
        result = "can get job"
    else:
        result = "Wont get a job"
        
    return render_template('index.html',prediction_value = result)


if __name__ == "__main__":
    app.run(debug=True)