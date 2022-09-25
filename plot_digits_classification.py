"""
================================
Recognizing hand-written digits
================================

This example shows how scikit-learn can be used to recognize images of
hand-written digits, from 0-9.

"""

# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
# License: BSD 3 clause

# Standard scientific Python imports
# import matplotlib.pyplot as plt
import itertools
import statistics

# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
from skimage.transform import rescale, resize, downscale_local_mean

# Model hyper-parameters
gamma_ = [0.001, 0.005, 0.01, 0.02, 0.04]
C_ = [0.1, 0.5, 1, 1.5, 2]

h_params = list(itertools.product(gamma_, C_))

###############################################################################
# Digits dataset
# --------------
#
# The digits dataset consists of 8x8
# pixel images of digits. The ``images`` attribute of the dataset stores
# 8x8 arrays of grayscale values for each image. We will use these arrays to
# visualize the first 4 images. The ``target`` attribute of the dataset stores
# the digit each image represents and this is included in the title of the 4
# plots below.
#
# Note: if we were working from image files (e.g., 'png' files), we would load
# them using :func:`matplotlib.pyplot.imread`.

digits = datasets.load_digits()

###############################################################################
# Classification
# --------------
#
# To apply a classifier on this data, we need to flatten the images, turning
# each 2-D array of grayscale values from shape ``(8, 8)`` into shape
# ``(64,)``. Subsequently, the entire dataset will be of shape
# ``(n_samples, n_features)``, where ``n_samples`` is the number of images and
# ``n_features`` is the total number of pixels in each image.
#
# We can then split the data into train and test subsets and fit a support
# vector classifier on the train samples. The fitted classifier can
# subsequently be used to predict the value of the digit for the samples
# in the test subset.

n_samples = len(digits.images)
IMAGE_SIZE = [(digits.images[0].shape[0], digits.images[0].shape[1])]
#, (digits.images[0].shape[0] // 2,
#                digits.images[0].shape[1] // 2), (6, 6), (16, 16)]

test_frac = 0.5
test_dev = 0.5
for image_size in IMAGE_SIZE:
    print(f"Image Size in digit dataset: {digits.images[0].shape}")
    print(f"Train_frac: {test_frac}, test_dev: {test_dev}")
    data = resize(digits.images, (len(digits.images), image_size[0], image_size[1]), anti_aliasing=True)
    # print(f"Image Size after transformation: {data[0].shape}")
    data = data.reshape((n_samples, -1))

    accuracy_list_all_params = []
    dev_acc_list = []
    train_acc_list = []
    test_acc_list = []
    best_acc = -1.0

    for h_param in h_params:
        accuracy_list = []
        # Create a classifier: a support vector classifier
        # flatten the images
        clf = svm.SVC(gamma=h_param[0], C=h_param[1])

        # Split data train and another set for dev and test
        X_train, X_split, y_train, y_split = train_test_split(
            data, digits.target, test_size=0.5, shuffle=False
        )

        X_dev, X_test, y_dev, y_test = train_test_split(
            X_split, y_split, test_size=0.5, shuffle=False
        )

        # Learn the digits on the train subset
        clf.fit(X_train, y_train)

        # Predict the value of the digit on the train subset
        predicted = clf.predict(X_train)
        train_acc = metrics.accuracy_score(y_train, predicted)
        accuracy_list.append(train_acc)
        train_acc_list.append(train_acc)

        # Predict the value of the digit on the dev subset
        predicted = clf.predict(X_dev)
        dev_acc = metrics.accuracy_score(y_dev, predicted)
        accuracy_list.append(dev_acc)
        dev_acc_list.append(dev_acc)

        # Predict the value of the digit on the test subset
        predicted = clf.predict(X_test)
        test_acc = metrics.accuracy_score(y_test, predicted)
        accuracy_list.append(test_acc)
        test_acc_list.append(dev_acc)

        ###############################################################################
        # :func:`~sklearn.metrics.classification_report` builds a text report showing
        # the main classification metrics.

        if test_acc > best_acc:
            best_acc = test_acc
            best_clf = clf
            report = metrics.classification_report(y_test, predicted)
            best_h_param = h_param

        accuracy_list_all_params.append(accuracy_list)

    # Tabular form of train dev and test accuracy
    print("h_param:                train  dev  test")
    for (h_param, i) in zip(h_params, accuracy_list_all_params):
        print("Gamma= {0:<5} C={1:<5}  : {2:.2f}  {3:.2f}  {4:.2f}".format(h_param[0], h_param[1], i[0], i[1], i[2]))

    # Print Max, min, mean and median of accuracies
    print("\n\n")
    print(
        f"Train: Max = {max(train_acc_list)}, Min = {min(train_acc_list)}, Median = {statistics.median(train_acc_list)}, "
        f"Mean = {statistics.mean(train_acc_list)}")
    print(
        f"Dev: Max = {max(dev_acc_list)}, Min = {min(dev_acc_list)}, Median = {statistics.median(dev_acc_list)}, "
        f"Mean = {statistics.mean(dev_acc_list)}")
    print(
        f"Test: Max = {max(test_acc_list)}, Min = {min(test_acc_list)}, Median = "
        f"{statistics.median(test_acc_list)}, Mean = {statistics.mean(test_acc_list)}")

    # Best model prediction
    print(f"\nClassification report with Image size {image_size} for best classifier {best_clf}: "
          f"{best_acc}\n")
