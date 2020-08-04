import random as r
import numpy as np
import matplotlib.pyplot as plt
import statistics 
import math
import scipy
def Adjusting_Files_Less_Than_Five():
    #this recreates the files without numbers 0,6,7,8,9
    try:
        train_file = open('ZipDigits.train', 'r')
        test_file = open('ZipDigits.test', 'r')
        
        adjusted_train = open('Adjusted.train', 'w')
        adjusted_test = open('Adjusted.test', 'w')
        
        count_lines_train = 0
        
        for line in train_file:
            numbers = line.split()
            count_lines_train += 1 
        
            for index, entry in enumerate(numbers):
                if index == 0:
                    digit = float(entry)
                if digit > 5 or digit == 0:
                    pass
                else:
                    adjusted_train.write(entry)
                    adjusted_train.write(" ")
                    if index == 256:
                        adjusted_train.write('\n')
        adjusted_train.close()
        
        for line in test_file:
            numbers = line.split()
            count_lines_train += 1 
        
            for index, entry in enumerate(numbers):
                if index == 0:
                    digit = float(entry)
                if digit > 5 or digit == 0:
                    pass
                else:
                    adjusted_test.write(entry)
                    if index == 256:
                        adjusted_test.write('\n')
        adjusted_test.close()
        print("successful file copy.")
    except:
        raise Exception("something failed copying files")
    pass
    
def Test_Plot():
    #Outputs 2 images of numbers, returns all numbers as 16x16 matrix
    matrices_plotted = []
    vec1 = np.empty((0,255),float,0)
    draw_file = open('Adjusted.train', 'r')
    for line_count, line in enumerate(draw_file):
            inp = [float(i) for i in line.split()]
            vec1 = np.concatenate([inp[1:]], axis = 0)
            digit = inp[0]
            #for some reason plt.imshow wants you to explicitly give it floats
            cleanout = np.reshape(vec1, (16,16))
            if line_count == 0 or line_count == 1:
                plt.imshow(cleanout)
                plt.show()
            data = [digit,cleanout]
            matrices_plotted.append(data)
    return matrices_plotted

def get_symmetry(matrix):
    image = matrix
    mirror_matrix = np.matrix([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                               [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                               [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],])

    imageflip = matrix.dot(mirror_matrix)
    imagesymm = abs(image - imageflip)
    symm_avg = -1 * get_avg_intensity(imagesymm)
    return symm_avg

def get_avg_intensity(matrix):
    avg_int = matrix.mean()
    return avg_int

def normalize(feature,mean,stddev):
    return ((feature * mean)/ stddev)

def cost_function(model, x, y):
    # Computes the cost function for all the training samples
    N = x.shape[0]
    total_cost = (1 / N) * np.sum(np.log(1 + math.exp((-y.dot(model)).dot(x))))
    return total_cost

def compute_gradient(x, y, target):
    # Computes the gradient of the cost function at the point theta

    return ((x * y) * (1 + math.exp(((-y * target) * x ))))


def gradient_descent(features, targets, step_size, max_iter = 500):
    w_current = np.zeros([features.shape[1],1]) 
    N = features.shape[0]
    outputs = []
    for i in range(max_iter):
           target = targets[i]
           if target == 5:
               sig = -1
           if target == 1:
               sig = 1
               pass
           if target != 5 and target != 1:
               i = i+1
           else:

           # probability = logistic_regression(sum(x * w_current))
            
          #  grad = compute_gradient(x,y, sig)
    
            # Update the model
       #     w_next = w_current - step_size * (features.T.dot) * ((logistic_regression(features.dot(w_current)) - targets))
            
            outputs.append(w_current)    
            w_current = w_next
    return outputs

def logistic_regression(w):
    return 1/ (1+ np.exp(-w))



if __name__ == '__main__':
    #Adjusting_Files_Less_Than_Five()
    all_matrices_training = Test_Plot()
    all_symm = []
    all_int = []
    digits = []

    
    for entry in all_matrices_training:
        digits.append(entry[0])
        matrix = entry[1]
        #Getting avg and stddev for intensities and symmetries
        intensity = get_avg_intensity(matrix)
        symmetry = get_symmetry(matrix)
        
        all_symm.append(symmetry)
        all_int.append(intensity)
    std_dev_int = statistics.stdev(all_int)
    std_dev_symm = statistics.stdev(all_symm)
    mean_int =  sum(all_int) / len(all_int)
    mean_symm = sum(all_symm) / len(all_symm)
        
    i = 0
    #normalizing and plotting intensities and symmetries
    while i < len(all_symm):
        digit = digits[i]
        intensities = all_int[i]
        symms = all_symm[i]
        new_int = normalize(intensities,mean_int,std_dev_int)
        new_Symm = normalize(symms, mean_symm,std_dev_symm)
        if digit == 1:
            plt.scatter(new_int,new_Symm, marker='o',color='b')
        if digit == 5:
            plt.scatter(new_int,new_Symm, marker = 'x', color = 'r')
        i += 1
    plt.show()
    
    
    #QUESTION 4 STUFF
    #implementing log regression/grad descent/ find seperator
    int_matrix = np.array(all_int)
    symm_matrix = np.array(all_symm)
    features = np.array([int_matrix,symm_matrix]).T
#    output = gradient_descent(features,digits, .1)
    
  #  for i in range(len(output)):
        
    #    plt.scatter(output[i][0],output[i][1])
    #somethigng wrong in caclulations 
    