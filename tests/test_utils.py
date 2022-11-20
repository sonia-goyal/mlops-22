import sys, os
import numpy as np
from joblib import load

sys.path.append(".")

from utils import get_all_h_param_comb, tune_and_save, h_param_tuning, train_dev_test_split, preprocess_digits
from sklearn import svm, metrics, datasets


# test case to check if all the combinations of the hyper parameters are indeed getting created
def test_get_h_param_comb():
    gamma_list = [0.01, 0.005, 0.001, 0.0005, 0.0001]
    c_list = [0.1, 0.2, 0.5, 0.7, 1, 2, 5, 7, 10]

    params = {"gamma": gamma_list, "C": c_list}
    h_param_comb = get_all_h_param_comb(params)

    assert len(h_param_comb) == len(gamma_list) * len(c_list)


def helper_h_params():
    # small number of h params
    gamma_list = [0.01, 0.005]
    c_list = [0.1, 0.2]

    params = {"gamma": gamma_list, "C": c_list}
    h_param_comb = get_all_h_param_comb(params)
    return h_param_comb


def helper_create_bin_data(n=100, d=7):
    x_train_0 = np.random.randn(n, d)
    x_train_1 = 1.5 + np.random.randn(n, d)
    x_train = np.vstack((x_train_0, x_train_1))
    y_train = np.zeros(2 * n)
    y_train[n:] = 1

    return x_train, y_train


def test_tune_and_save():
    h_param_comb = helper_h_params()
    x_train, y_train = helper_create_bin_data(n=100, d=7)
    x_dev, y_dev = x_train, y_train

    clf = svm.SVC()
    metric = metrics.accuracy_score

    model_path = "test_run_model_path.joblib"
    actual_model_path = tune_and_save(clf, x_train, y_train, x_dev, y_dev, metric, h_param_comb, model_path)

    assert actual_model_path == model_path
    assert os.path.exists(actual_model_path)
    assert type(load(actual_model_path)) == type(clf)


def countElement(sample_list, element):
    return sample_list.count(element)


def test_model_bias():
    h_param_comb = helper_h_params()
    x_train, y_train = helper_create_bin_data(n=100, d=7)

    x_train, y_train, x_dev, y_dev, x_test, y_test = train_dev_test_split(
        x_train, y_train, 0.8, .1
    )

    clf = svm.SVC()
    metric = metrics.accuracy_score

    best_model, best_metric, best_h_params = h_param_tuning(h_param_comb, clf, x_train, y_train, x_dev, y_dev, metric)
    pred = best_model.predict(x_test)

    pred = list(pred)
    store_count = []
    classes = set(pred)
    for cls in classes:
        count = countElement(pred, cls)
        store_count.append(count)

    print(store_count)
    print("======= Hello World =======")
    sum_count = sum(store_count)
    for i in store_count:
        per = i / sum_count
        assert per <= 50


def test_predict_all_classes():
    h_param_comb = helper_h_params()
    x_train, y_train = helper_create_bin_data(n=100, d=7)

    ground_truth_labels = set(y_train)
    x_train, y_train, x_dev, y_dev, x_test, y_test = train_dev_test_split(
        x_train, y_train, 0.8, .1
    )

    clf = svm.SVC()
    metric = metrics.accuracy_score

    best_model, best_metric, best_h_params = h_param_tuning(h_param_comb, clf, x_train, y_train, x_dev, y_dev, metric)
    pred = best_model.predict(x_test)

    classes = set(pred)
    assert ground_truth_labels == classes  # classes.issubset(ground_truth_labels) == True


def test_randomness_of_splits():
    h_param_comb = helper_h_params()
    x_train, y_train = helper_create_bin_data(n=100, d=7)

    x_train, y_train, x_dev, y_dev, x_test, y_test = train_dev_test_split(
        x_train, y_train, 0.8, .1, 42
    )

    x_train1, y_train1, x_dev1, y_dev1, x_test1, y_test1 = train_dev_test_split(
        x_train, y_train, 0.8, .1, 42
    )


    assert np.array_equal(x_train, x_train1)


def test_randomness_of_splits2():
    h_param_comb = helper_h_params()
    x_train, y_train = helper_create_bin_data(n=100, d=7)

    x_train, y_train, x_dev, y_dev, x_test, y_test = train_dev_test_split(
        x_train, y_train, 0.8, .1, 10
    )

    x_train1, y_train1, x_dev1, y_dev1, x_test1, y_test1 = train_dev_test_split(
        x_train, y_train, 0.8, .1, 42
    )


    assert np.array_equal(x_train, x_train1)
