# CSE 404 Introduction to Machine Learning
# Python Demo for Gradient Descent
#

import numpy as np
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

def compute_gradient(feature, target,model):
    # Compute the gradient of linear regression objective function with respect to w
    x,t,w = feature,target,model
    
    grad = x.T.dot(x.dot(w)) - (x.T.dot(t))
    return grad 

def compute_objective_value(feature, target, model):
    pred_value = np.matmul(feature, model)
    # Compute MSE 
    t = target
    difference = pred_value - t
    return difference.T.dot(difference)[0][0]


def gradient_descent(feature, target, step_size, max_iter):
    w_current = np.zeros([feature.shape[1],1])
    objective_value = []
    for i in range(max_iter):
        # Compute gradient
        grad = compute_gradient(feature, target, w_current)

        # Update the model
        w_next = w_current - step_size * (grad)

        # Compute the error (objective value)
        objective_value.append(compute_objective_value(feature, target, w_next))
        if i > 0:
            if abs(objective_value[i] - objective_value[i-1]) < 1e-5:
                print("Converged!")
                break
        w_current = w_next
    return w_current, np.array(objective_value)


def plot_objective_function(objective_value):

    plt.figure()
    plt.plot(objective_value)
    plt.xlabel("Number of iterations")
    plt.ylabel("Objective value")
    plt.title("Convergence of least squares with gradient descent")
    plt.show()


def generate_data(sample_size,feature_size,train_perc):
    data = np.concatenate((np.random.randn(sample_size, feature_size), np.ones((sample_size, 1))), axis=1) 

    # Generate ground truth model.
    truth_model = np.random.randn(feature_size + 1, 1) * 10 

    # Generate label.
    label = np.dot(data, truth_model)

    # add element-wise gaussian noise to each label.
    label += np.random.randn(sample_size, 1)

    num_train_sample = np.max([np.floor(sample_size * train_perc).astype(int), 1])
    data, label = shuffle(data, label)

    train_data = data[:num_train_sample]
    test_data = data[num_train_sample:]

    train_label = label[:num_train_sample]
    test_label = label[num_train_sample:]

    return train_data, test_data, train_label, test_label


def print_train_test_error(train_data, test_data, train_label, test_label, model):
    train_error = compute_objective_value(train_data, train_label, model)
    test_error = compute_objective_value(test_data, test_label, model)

    print("Train error: %f"%(train_error))
    print("Test error: %f"%(test_error))



if __name__ == '__main__':
    plt.interactive(False)

    np.random.seed(491)

    train_data, test_data, train_label, test_label = generate_data(500, 50, 0.7)

    model, obj_value = gradient_descent(train_data, train_label, 0.001, 1000)

    plot_objective_function(obj_value)

    print_train_test_error(train_data, test_data, train_label, test_label, model)



   
