import pickle 
import os
from flask import Flask, request, app, jsonify, url_for, render_template
import pandas as pd 
import numpy as np

# instantiate the app
app = Flask(__name__)

# load the model
model = pickle.load(open("model_1.pkl", "rb"))

# load the scaler
scaler = pickle.load(open("scaler.pkl", "rb"))

# home page route
@app.route("/")
def home_page():
    return render_template("home.html")

@app.route('/docs')
def documentation():
    return render_template("documentation.html")

# prediction route
@app.route('/predict', methods=['POST'])
def predict():
    # get the data from the post request
    data = request.json['data']
    # convert data into array
    data = np.array(list(data.values())).reshape(1, -1)
    # scale the data
    data = scaler.transform(data)
    # make prediction
    prediction = model.predict(data)
    # return the prediction as json
    return jsonify({'prediction': prediction[0]})


if __name__ == '__main__': 
    app.run(port=5000, debug=True)
