# CSE 404 Introduction to Machine Learning
# Python Lab for Linear Regression.
#
# By Jiayu Zhou, 2019

import time
import numpy as np
from sklearn.utils import shuffle
import matplotlib.pyplot as plt


def rand_split_train_test(data, label, train_perc):
    if train_perc >= 1 or train_perc <= 0:
        raise Exception('train_perc should be between (0,1).')
    sample_size = data.shape[0]
    if sample_size < 2:
        raise Exception('Sample size should be larger than 1. ')

    num_train_sample = np.max([np.floor(sample_size * train_perc).astype(int), 1])
    data, label = shuffle(data, label)

    data_tr = data[:num_train_sample]
    data_te = data[num_train_sample:]

    label_tr = label[:num_train_sample]
    label_te = label[num_train_sample:]

    return data_tr, data_te, label_tr, label_te

#x = data, t = label
def subsample_data(data, label, subsample_size):
    # protected sample size
    subsample_size = np.max([1, np.min([data.shape[0], subsample_size])])

    data, label = shuffle(data, label)
    data = data[:subsample_size]
    label = label[:subsample_size]
    return data, label


def generate_rnd_data(feature_size, sample_size, bias=False):

    # Generate X matrix.
    print("TEST: ",np.random.randn(sample_size,feature_size).shape)
    data = np.concatenate((np.random.randn(sample_size, feature_size), np.ones((sample_size, 1))), axis=1) \
        if bias else np.random.randn(sample_size, feature_size)  # the first dimension is sample_size (n X d)
    print("data shape: ", data.shape)
    print(data)
    # Generate ground truth model.
    truth_model = np.random.randn(feature_size + 1, 1) * 10 \
        if bias else np.random.randn(feature_size, 1) * 10
    print("TM: ", truth_model)
    print("truth model: ", truth_model.shape)
    # Generate label.
    label = np.dot(data, truth_model)

    # add element-wise gaussian noise to each label.
    label += np.random.randn(sample_size, 1)
    
    print("label: ", label.shape)
    print(label)
    return data, label, truth_model


def mean_squared_error(true_label, predicted_label):
    """
        Compute the mean square error between the true and predicted labels
        :param true_label: Nx1 vector
        :param predicted_label: Nx1 vector
        :return: scalar MSE value
    """
    mse = np.sqrt(np.sum((true_label - predicted_label)**2)/true_label.size)
    return mse

def least_squares(feature, target):
    """
    Compute least squares using closed form
    :param feature: X
    :param target: y
    :return: computed weight vector
    """
    x = feature
    t = target
    w_star = np.linalg.inv((x.T.dot(x))).dot(x.T.dot(t))

    # TODO: Compute the model of least squares.
    return w_star


def ridge_regression(feature, target, lam=1e-17):
    """
    Compute ridge regression using closed form
    :param feature: X
    :param target: y
    :param lam: lambda
    :return:
    """
    feature_dim = feature.shape[1]
    x = feature
    t = target
    w_star = np.linalg.inv((x.T.dot(x) + np.eye(feature_dim).dot(lam))).dot(x.T.dot(t))

    # TODO: Compute the model of ridge regression. 
    return w_star




def exp1():
    # EXP1: training testing.
    # generate a data set.
    (feature_all, target_all, model) = generate_rnd_data(feature_size=3, sample_size=20, bias=False)
    # split training/testing
    feature_train, feature_test, target_train, target_test = rand_split_train_test(feature_all, target_all, train_perc=0.8)
    # compute model
    reg_model_lsqr = least_squares(feature_train, target_train)
    reg_model_ridge = ridge_regression(feature_train, target_train, lam=1e-7)

    # evaluate performance
    print('Training MSE(lsqr):', mean_squared_error(target_train, np.dot(feature_train, reg_model_lsqr)))
    print('Testing MSE(lsqr):', mean_squared_error(target_test, np.dot(feature_test, reg_model_lsqr)))
    print('Training MSE(ridge):', mean_squared_error(target_train, np.dot(feature_train, reg_model_ridge)))
    print('Testing MSE(ridge):', mean_squared_error(target_test, np.dot(feature_test, reg_model_ridge)))


def exp2():
    # EXP2: generalization performance: increase sample size.
    different_sample_sizes = [50, 100, 150, 200, 250, 300, 350, 400, 450]
    (feature_all, target_all, model) = generate_rnd_data(feature_size=100, sample_size=1000, bias=True)
    feature_hold, feature_test, target_hold, target_test = \
        rand_split_train_test(feature_all, target_all, train_perc=0.9)

    train_performance = []
    test_performance = []
    for train_sample_size in different_sample_sizes:
        feature_train, target_train = subsample_data(feature_hold, target_hold, train_sample_size)
        reg_model = ridge_regression(feature_train, target_train, lam=1e-5)
        train_performance += [mean_squared_error(target_train, np.dot(feature_train, reg_model))]
        test_performance += [mean_squared_error(target_test, np.dot(feature_test, reg_model))]

    print(train_performance)
    print(test_performance)

    plt.figure()
    train_plot, = plt.plot(different_sample_sizes, np.log10(train_performance), linestyle='-', color='b',
                           label='Training Error')
    test_plot, = plt.plot(different_sample_sizes, np.log10(test_performance), linestyle='-', color='r', label='Testing '
                                                                                                         'Error')
    plt.xlabel("Sample Size")
    plt.ylabel("Error (log)")
    plt.title("Generalization performance: increase sample size fix dimensionality")
    plt.legend(handles=[train_plot, test_plot])
    plt.show()


def exp3():
    # EXP3: generalization performance: increase dimensionality.
    different_dimensionality = [100, 150, 200, 250, 300, 350, 400, 450]

    train_performance = []
    test_performance = []
    for dimension in different_dimensionality:
        (feature_all, target_all, model) = generate_rnd_data(feature_size=dimension, sample_size=1000, bias=True)
        feature_train, feature_test, target_train, target_test = \
            rand_split_train_test(feature_all, target_all, train_perc=0.9)
        reg_model = ridge_regression(feature_train, target_train, lam=1e-5)
        train_performance += [mean_squared_error(target_train, np.dot(feature_train, reg_model))]
        test_performance += [mean_squared_error(target_test, np.dot(feature_test, reg_model))]

    print(train_performance)
    print(test_performance)

    plt.figure()
    train_plot, = plt.plot(different_dimensionality, np.log10(train_performance), linestyle='-', color='b',
                           label='Training Error')
    test_plot, = plt.plot(different_dimensionality, np.log10(test_performance), linestyle='-', color='r', label='Testing '
                                                                                                         'Error')
    plt.xlabel("Dimensionality")
    plt.ylabel("Error (log)")
    plt.title("Generalization performance: increase dimensionality fix sample size")
    plt.legend(handles=[train_plot, test_plot])
    plt.show()


def exp4():
    # EXP4: computational time: increase dimensionality.
    different_dimensionality = range(100, 2000, 100)

    train_performance = []
    test_performance = []
    time_elapse = []
    for dimension in different_dimensionality:
        (feature_all, target_all, model) = generate_rnd_data(feature_size=dimension, sample_size=1000, bias=True)
        feature_train, feature_test, target_train, target_test = \
            rand_split_train_test(feature_all, target_all, train_perc=0.9)
        t = time.time()
        reg_model = ridge_regression(feature_train, target_train, lam=1e-5)
        time_elapse += [time.time() - t]
        print('Finished model of dimension {}'.format(dimension))

        train_performance += [mean_squared_error(target_train, np.dot(feature_train, reg_model))]
        test_performance += [mean_squared_error(target_test, np.dot(feature_test, reg_model))]

    plt.figure()
    time_plot, = plt.plot(different_dimensionality, time_elapse, linestyle='-', color='r', label='Time cost')
    plt.xlabel("Dimensionality")
    plt.ylabel("Time (ms)")
    plt.title("Computational efficiency.")
    plt.legend(handles=[time_plot])
    plt.show()


if __name__ == '__main__':
    plt.interactive(False)

    # set seeds to get repeatable results.
    np.random.seed(491)

    # EXP1: training testing.
    exp1()
    #
    # # EXP2: generalization performance: increase sample size.
   # exp2()
    #
    # # EXP3: generalization performance: increase dimensionality.
  #  exp3()
    #
    # # EXP4: computational complexity by varing dimensions.
   # exp4()


