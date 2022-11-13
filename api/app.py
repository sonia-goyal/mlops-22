from flask import Flask
from flask import jsonify
from flask import request
from joblib import dump, load

app = Flask(__name__)
model_path = "/Users/soniagoyal/mlops/mlops-22/svm_gamma=0.001_C=0.5.joblib"
model = load(model_path)

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"


@app.route("/sum", methods=["POST"])
def sum():
    x = request.json['x']
    y = request.json['y']
    z = x + y
    return {'sum': z}


@app.route("/predict", methods=['POST'])
def predict_digit():
    image = request.json['image']
    print("done loading")
    predicted = model.predict([image])
    return {"y_predicted": int(predicted[0])}

