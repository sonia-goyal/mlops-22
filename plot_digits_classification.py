# PART: library dependencies -- sklear, torch, tensorflow, numpy, transformers

# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
import pdb
from sklearn import tree

from utils import (
    preprocess_digits,
    train_dev_test_split,
    h_param_tuning,
    h_param_tuning_dec,
    data_viz,
    pred_image_viz,
    get_all_h_param_comb,
    tune_and_save,
    predict
)
from joblib import dump, load

train_frac, dev_frac, test_frac = 0.8, 0.1, 0.1
assert train_frac + dev_frac + test_frac == 1.0

# 1. set the ranges of hyper parameters
gamma_list = [0.01, 0.005, 0.001, 0.0005, 0.0001]
c_list = [0.1, 0.2, 0.5, 0.7, 1, 2, 5, 7, 10]

params = {"gamma": gamma_list, "C": c_list}

h_param_comb = get_all_h_param_comb(params)


# PART: load dataset -- data from csv, tsv, jsonl, pickle
digits = datasets.load_digits()
data_viz(digits)
data, label = preprocess_digits(digits)
# housekeeping
del digits

#
# x_train, y_train, x_dev, y_dev, x_test, y_test = train_dev_test_split(
#     data, label, train_frac, dev_frac
# )


# def create_split():
#     x_train, y_train, x_dev, y_dev, x_test, y_test = train_dev_test_split(
#         data, label, train_frac, dev_frac
#     )




models = [[tree.DecisionTreeClassifier(), 'decision_tree'], [svm.SVC(), 'svm']]
perf_test = {}
metric = metrics.accuracy_score
for model in models:
    for i in range(5):
        print(f"\nTraining for split: {i+1}")
        x_train, y_train, x_dev, y_dev, x_test, y_test = train_dev_test_split(data, label, train_frac, dev_frac)
        clf = model[0]
        if model[1] == 'svm':
            best_model, best_metric, best_h_params = h_param_tuning(h_param_comb, clf, x_train, y_train, x_dev, y_dev, metric)
        else:
            best_model, best_metric, best_h_params = h_param_tuning_dec(clf, x_train, y_train, x_dev, y_dev, metric)
        perf_test[model[1] + '_' + str(i)] = predict(best_model, x_test, y_test, metric)


print(perf_test)

# actual_model_path = tune_and_save(
#     clf, x_train, y_train, x_dev, y_dev, metric, h_param_comb, model_path=None
# )
#
#
# # 2. load the best_model
# best_model = load(actual_model_path)
#
# # PART: Get test set predictions
# # Predict the value of the digit on the test subset
# predicted = best_model.predict(x_test)
#
# pred_image_viz(x_test, predicted)

# 4. report the test set accurancy with that best model.
# PART: Compute evaluation metrics
# print(
#     f"Classification report for classifier {clf}:\n"
#     f"{metrics.classification_report(y_test, predicted)}\n"
# )