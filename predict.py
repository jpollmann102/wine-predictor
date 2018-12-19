###############################################################################
# I originally wrote most of commented code as practice just to better        #
# understand the inner workings of these algorithms                           #
# I switched to using scikit to learn how to use that and just for simplicity #
# Note that this particular file is NOT completed                             #
###############################################################################

# import sys
# import numpy as np
# import pandas as pd
#
# from math import exp
#
# if(len(sys.argv) < 2):
#     print("Please enter the name of the file you wish to analyze");
#     sys.exit()
#
# filename = sys.argv[1]
#
# # read in file
# file = pd.read_csv('test/' + filename, header=None)
#
# # read in thetas
# theta1 = pd.read_csv('models/theta1.csv', header=None)
# theta2 = pd.read_csv('models/theta2.csv', header=None)
# theta3 = pd.read_csv('models/theta3.csv', header=None)
#
# # read in mu and sigma for feature scaling
# mu = pd.read_csv('models/mu.csv', header=None)
# mu = mu.T
# sigma = pd.read_csv('models/sigma.csv', header=None)
# sigma = sigma.T
#
# # get number of features
# n = len(mu.columns)
#
# # function for computing sigmoid
# def sigmoid(x):
#     return 1 / (1 + exp(-x))
#
# # function to scale the values in the user's file
# def normalize(X):
#     normalized = np.subtract(X, mu)
#     normalized = np.divide(normalized, sigma)
#
#     return normalized
#
# normalized = normalize(file)
#
# def predict(X, theta, threshold=0.5):
#     m = len(X.index)
#     p = np.zeros(m)
#
#     # apply sigmoid to values
#     # h = np.multiply(theta.iloc[1:,].T, X)
#     # h = theta.iloc[1:,].T * X
#     # h = np.matmul(theta.iloc[1:,].T, X)
#     # h = X * theta.T
#     print(h)
#     h = h.applymap(sigmoid)
#     print(h)
#
#     for i in range(m):
#         p[i] = 1 if h[i] >= threshold else 0
#
#     return p
#
# p = predict(normalized, theta1)
# print(p)
