import random as r
import numpy as np
import matplotlib.pyplot as plt
import statistics 
import scipy.io
from operator import itemgetter

def Test_Plot(vector):
    #output 256 length vector to number
    try:
        cleanout = np.reshape(vector, (16,16))
        plt.imshow(cleanout)
        plt.show()
    except:
        print(f"Failed to output Vector")
        return False
    
    return True

#loading matlab files in python
data = scipy.io.loadmat("data.MAT")["X"]
data_labels = scipy.io.loadmat("data.MAT")["Y"]

usps = scipy.io.loadmat("USPS.MAT")["A"]
usps_labels = scipy.io.loadmat("USPS.MAT")["L"]

#this is our data, and the mean of it
x_input = usps
x_standardized = x_input - np.mean(x_input)

#create covariance matrix
cov_x = x_standardized.T.dot(x_standardized)

#organizing eigenvalues/vectors
val_vect = []
value , vector = np.linalg.eig(cov_x)
for i in range(0, len(value)):
     temp_tuple = (value[i] , vector[i])
     val_vect.append(temp_tuple)
    
#sorting eigen values/vectors greatest to smallest by eigen value
val_vect.sort(key=itemgetter(0) , reverse = True)
sorted_vectors = []
for item in val_vect:
    sorted_vectors.append(item[1])
    
#principle components
d_values = [10 , 50 , 100, 200, 400]

for component_amount in d_values:
    #G is 256 by d_values
    g_matrix = np.row_stack(sorted_vectors[0:component_amount] )
    reconstructed = []
    for i in range(0, len(x_standardized)):
        c = g_matrix.dot(x_standardized[i])
        x_hat = g_matrix.T.dot(c) + np.mean(x_input)
        reconstructed.append(x_hat)

    #calculate error
    x_recon = np.asarray(reconstructed)
    error = np.linalg.norm((x_input - (x_recon + np.mean(x_input))), 'fro')
    total_error = 100*error/(len(x_standardized)*len(x_standardized[1]))
    print(f"Total Error after {component_amount} feature reconstuction : {total_error}")
    #output image
    for i in range(0,2):
        print("Original")
        Test_Plot(x_standardized[i])
        print("reconstructed")
        Test_Plot(reconstructed[i])
        
    