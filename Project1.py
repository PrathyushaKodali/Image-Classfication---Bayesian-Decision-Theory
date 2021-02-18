# -*- coding: utf-8 -*-


import scipy.io as sc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# LOAD TRAIN FILES
Numpyfile_0 =sc.loadmat('training_data_0.mat')
Numpyfile_1 =sc.loadmat('training_data_1.mat')

# LOAD TEST FILES
Numpyfile_test_0=sc.loadmat('testing_data_0.mat')
Numpyfile_test_1=sc.loadmat('testing_data_1.mat')

# CONVERT TO 784 - DIMENSIONAL VECTOR
def convert_to_vector(Numpyfile_0,Numpyfile_1):
    Images_0 = []
    for i in range(len(Numpyfile_0['nim0'][0][0])):
        x = np.array(Numpyfile_0.get('nim0')[0:28,0:28,i])
        Images_0.append(x.flatten())

    Images_1 = []
    for i in range(len(Numpyfile_1['nim1'][0][0])):
        x= np.array(Numpyfile_1.get('nim1')[0:28,0:28,i])
        Images_1.append(x.flatten())

    # CONCATENATE BOTH FILES
    length_of_class_0 = len(Images_0)
    length_of_class_1 = len(Images_1)
    total_length = length_of_class_0  +length_of_class_1
    Images_merged = np.concatenate((Images_0, Images_1))
    df_image = pd.DataFrame(Images_merged)
    return df_image,length_of_class_0,total_length

df_image, length_train_split,tot_len_train = convert_to_vector(Numpyfile_0,Numpyfile_1)
df_test_image, length_test_split,tot_len_test = convert_to_vector(Numpyfile_test_0,Numpyfile_test_1)
print("Train & Test datasets converted to 784 dimensional vector ")

#****************************************************************
#      TASK-1: FEATURE - NORMALIZATION                          *
#****************************************************************
print(" Feature Normalization ")
def normalize(df):
    normalized_df=(df-df.mean())/df.std()
    return normalized_df

df_norm = normalize(df_image)
df_test_norm = normalize(df_test_image)

#****************************************************************
#                  TASK-2: PCA                                  *
#****************************************************************
print(" PCA ")
def PCA(normalized_df):
    # Calculating the covariance matrix
    covariance_matrix = np.cov(normalized_df.T)
    # Using np.linalg.eig function
    eigen_values, eigen_vectors = np.linalg.eig(covariance_matrix)
    print("Eigenvector: \n",eigen_vectors,"\n")
    print("Eigenvalues: \n", eigen_values, "\n")

    # Calculating the explained variance on each of components
    variance_explained = []
    for i in eigen_values:
        variance_explained.append((i / sum(eigen_values)) * 100)
    print("Explained variance \n",variance_explained)

    cumulative_variance_explained = np.cumsum(variance_explained)

    pc1_pc2 = (eigen_vectors.T[:][:2]).T
    df_PCA = normalized_df.dot(pc1_pc2)
    return df_PCA

df_train_pca = PCA(df_norm)
df_test_pca = PCA(df_test_norm)
#****************************************************************
#              TASK-3: DIMENSION REDUCTION                      *
#****************************************************************
print(" Plotting the data samples ")
def PCA_Plot(X_pca,length,tot_length,title):
    for i in range(0,length):
        plt.scatter(X_pca.iloc[i,0], X_pca.iloc[i,1], c='blue', label = 'Image0')
    for i in range(length,tot_length):
        plt.scatter(X_pca.iloc[i,0], X_pca.iloc[i,1], c='red', label = 'Image1')
    plt.xlabel('Principal Component-1')
    plt.ylabel('Principal Component-2')
    plt.title(title)
    plt.show()

PCA_Plot(df_train_pca,length_train_split,tot_len_train,"train")
PCA_Plot(df_test_pca,length_test_split,tot_len_test,"test")

#***************************************************************#
#              TASK - 4 : PARAMETER ESTIMATION                  #
#***************************************************************#

X_matrix = df_train_pca.to_numpy()
X_new = np.split(X_matrix, [length_train_split])
xmatrix_0 = X_new[0]
xmatrix_1 = X_new[1]

# Mean of Class 0 and Class 1
mu0=(np.mean(xmatrix_0,axis=0))
mu1=(np.mean(xmatrix_1,axis=0))

mu0=mu0[:,None]
mu1=mu1[:,None]

# Variance of Class 0 and Class 1
sigma0=np.cov(xmatrix_0, rowvar = False, bias = False)
sigma1=np.cov(xmatrix_1, rowvar = False, bias = False)


print("################# Parameters of Digit 0 #####################")
print("Mean vector=", mu0)
print("Covariance Matrix=", sigma0)
print("################# Parameters of Digit 1 #####################")
print("Mean vector=", mu1)
print("Covariance Matrix=", sigma1)
print("######################################")

#*********************************************************************************************#
#              TASK - 5 : Bayesian Decision Theory for optimal classification                 #
#*********************************************************************************************#

def decisionboundary(mu0, mu1, sigma0, sigma1, x):
    class_conditional_image0 = np.exp(
        -0.5 * (np.matmul(np.matmul((x - mu0).transpose(), np.linalg.inv(sigma0)), (x - mu0)))) / (
                                           np.sqrt(np.linalg.det(sigma0)) * 2 * np.pi)
    class_conditional_image1 = np.exp(
        -0.5 * (np.matmul(np.matmul((x - mu1).transpose(), np.linalg.inv(sigma1)), (x - mu1)))) / (
                                           np.sqrt(np.linalg.det(sigma1)) * 2 * np.pi)
    return class_conditional_image0[0][0] / class_conditional_image1[0][0]


def errorcal(imgarray, true, mu0, mu1, sigma0, sigma1):
    length = len(imgarray)
    errorcount = 0
    for i in range(length):
        x = imgarray[i][:, None]
        if true == 0:
            if decisionboundary(mu0, mu1, sigma0, sigma1, x) < 1:
                errorcount += 1
        if true == 1:
            if decisionboundary(mu0, mu1, sigma0, sigma1, x) > 1:
                errorcount += 1
    return errorcount


# Calculating train error
training_error=errorcal(xmatrix_0,0,mu0,mu1,sigma0,sigma1)+errorcal(xmatrix_1,1,mu0,mu1,sigma0,sigma1)

# Split the merged test PCA dataset for two classes
X_test_matrix = df_test_pca.to_numpy()
X_test_new = np.split(X_test_matrix, [length_test_split])
xmatrix_test_0= X_test_new[0]
xmatrix_test_1 = X_test_new[1]

# Calculating Testing error
test_error=errorcal(xmatrix_test_0,0,mu0,mu1,sigma0,sigma1)+errorcal(xmatrix_test_1,1,mu0,mu1,sigma0,sigma1)

print("training data accuracy=",(1-((training_error)/((len(xmatrix_0)+len(xmatrix_1)))))*100,"%")

print("test data accuracy=",(1-((test_error)/(len(xmatrix_test_0)+len(xmatrix_test_1))))*100,"%")
