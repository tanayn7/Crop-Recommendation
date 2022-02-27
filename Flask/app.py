import pickle
from distutils.log import debug
import numpy
from flask import Flask, render_template, request, jsonify

#create flask app
app = Flask(__name__)

#load the model
model = pickle.load(open("model.pkl", "rb"))


@app.route("/")
def Home():
    return render_template('index.html')


@app.route("/predict", methods = ["POST"])
def predict():
    all_features = [float(x) for x in request.form.values()]
    #print(all_features)
    features = [numpy.array(all_features)]
    prediction = model.predict(features)
    return render_template("index.html", prediction_text = "Recommended Crop is  {}".format("prediction")) 



if __name__ == "__main__":
    app.run(debug=True)