###############################################################################
# I originally wrote most of commented code as practice just to better        #
# understand the inner workings of these algorithms                           #
# I switched to using scikit to learn how to use that and just for simplicity #
###############################################################################

import pandas as pd
# import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
# from math import exp

# read in data
df = pd.read_csv('data/wines.csv', header=None)

# subtract by 1 to account for results column
n = len(df.columns)
# m = len(df.index)

# extract results column
y = df.iloc[:,0]

# create arrays for one vs all hypotheses
# y1 = y.where(y == 1, 0)
# y2 = y.where(y == 2, 0)
# y2 = y2.where(y2 == 0, 1)
# y3 = y.where(y == 3, 0)
# y3 = y3.where(y3 == 0, 1)

# extract features
X = df.iloc[:,1:n]

xTrain, xTest, yTrain, yTest = train_test_split(X, y, test_size=0.25, random_state=0)
# logreg = LogisticRegression(C=1e5, solver='lbfgs', multi_class='multinomial')
logreg = LogisticRegression()
logreg.fit(xTrain, yTrain)
predictions = logreg.predict(xTest)
print("\n{}".format(predictions))
score = logreg.score(xTest, yTest)
print("\nAccuracy was: {:.2f}%".format(score * 100))

# # function for scaling values
# def normalize(X):
#     Xnorm = X
#     mu = np.zeros(n - 1)
#     sigma = np.zeros(n - 1)
#
#     for i in range(0,n - 1):
#         mu[i] = np.mean(X.iloc[:,i])
#         sigma[i] = np.std(X.iloc[:,i])
#
#         Xnorm.iloc[:,i] = np.subtract(X.iloc[:,i], mu[i])
#         Xnorm.iloc[:,i] = Xnorm.iloc[:,i] / sigma[i]
#
#     return Xnorm, mu, sigma
#
# # normalize values in X
# X, mu, sigma = normalize(X)
#
# # add intercept term
# X.insert(loc=0, column='0', value=np.ones(m))
#
# # function for computing sigmoid
# def sigmoid(x):
#     return 1 / (1 + exp(-x))
#
# # function for determing cost given a theta set
# def cost(X, y, theta):
#     m = y.size
#     J = 0
#
#     divisor = (1 / m) * -1
#
#     h = X * theta
#     h = h.applymap(sigmoid)
#
#     for i in range(0,m):
#         J += (y[i] * np.log(h.iloc[i,:])) + ((1 - y[i]) * np.log(1 - h.iloc[i,:]))
#
#     J *= divisor
#
#     return J
#
# # function for logistic regression gradient descent
# def gradientDescent(X, y, theta, step, iters, output=False):
#
#     for iter in range(0,iters):
#         tempTheta = theta
#
#         h = X * theta
#         h = h.applymap(sigmoid)
#
#         for i in range(0, n):
#             difference = h.iloc[:,i].sub(y, axis=0)
#             tempProduct = np.multiply(difference, X.iloc[:,i])
#             tempSum = np.sum(tempProduct)
#             tempTheta[i] = theta[i] - (step * tempSum)
#
#         theta = tempTheta
#
#         if output:
#             if iter % 50 == 0:
#                 print(cost(X, y, theta))
#
#     return theta
#
# # set up gradient descent parameters
# alpha = 0.01
# iterations = 400
#
# theta1 = np.zeros(n)
# theta2 = np.zeros(n)
# theta3 = np.zeros(n)
#
# # calculate thetas for each class
# theta1 = gradientDescent(X, y1, theta1, alpha, iterations)
# theta2 = gradientDescent(X, y2, theta2, alpha, iterations)
# theta3 = gradientDescent(X, y3, theta3, alpha, iterations)
#
# # export models
# np.savetxt('models/theta1.csv', theta1, delimiter=",")
# np.savetxt('models/theta2.csv', theta2, delimiter=",")
# np.savetxt('models/theta3.csv', theta3, delimiter=",")
# np.savetxt('models/mu.csv', mu, delimiter=",")
# np.savetxt('models/sigma.csv', sigma, delimiter=",")
