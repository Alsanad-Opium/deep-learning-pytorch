from flask import Flask,request,jsonify, render_template
import numpy as np 
import pickle

app = Flask(__name__) # initializing a flask app


## laoding the models 
model = pickle.load(open("placement_model.pkl","rb")) 
scalar = pickle.load(open("scalar.pkl", "rb"))



@app.route('/') ## route to display the home page app.route is a decorator 

def home():
    return render_template('index.html')


@app.route('/predict',methods = ['POST']) ## this is import for routing  the POSt request that will be sent fro the index.html
## form will send the data via the action attribute of the form 
def predict():
    
    cgpa = float(request.form['cgpa']) ## fetching the data from the form
    iq = float(request.form['iq'])
    
    input_data = np.array([[cgpa,iq]]) # converting the data into numpy array as we 
    # are using the model which was trained on numpy array 
    
     ## scaling the input data
    scaled_data = scalar.transform(input_data) ## as the model was trained on scaled data therefore 
    # we have to scale the input aswell 
    
    prediction = model.predict(scaled_data)[0]  ## making the predictio aand as the it is by default predicted as the numpy array therefore we are fetching the first value
    
    if prediction == 1:
        result = "can get job"
    else:
        result = "Wont get a job"
        
    return render_template('index.html',prediction_value = result) ## returning the result to the index.html page ## this will refresh the page 


if __name__ == "__main__":
    app.run(debug=True) 