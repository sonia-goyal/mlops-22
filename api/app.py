from flask import Flask
from flask import jsonify
from flask import request
import sys
from joblib import load

sys.path.append(".")

app = Flask(__name__)
# model_path = "/Users/soniagoyal/mlops/mlops-22/" + "svm_gamma=0.001_C=0.5.joblib"
# model = load(model_path)


@app.route("/")
def hello_world():
    return "<b><p>Hello, World!</p><b>"


@app.route("/sum", methods=["POST"])
def sum():
    x = request.json['x']
    y = request.json['y']
    z = x + y
    return {'sum': z}


@app.route("/predict", methods=['POST'])
def predict_digit():
    image = request.json['image']
    model = request.json['model']
    if model == 'svm':
        model_ = 'svm_gamma=0.001_C=0.5.joblib'
    else:
        model_ = 'decision_tree_decision_tree_entropy.joblib'

    model_path = "/exp/models/" + model_
    model = load(model_path)

    print("done loading")
    predicted = model.predict([image])
    return {"y_predicted": int(predicted[0])}


# @app.route("/predict", methods=['POST'])
# def predict_digit():
#     image1 = request.json['image1']
#     image2 = request.json['image2']
#     print("done loading")
#     predicted1 = model.predict([image1])
#     predicted2 = model.predict([image2])
#     if int(predicted1[0]) == int(predicted2[0]):
#         val = 'Same number'
#     else:
#         val = 'Different numbers'
#     return {"prediction": val}

