# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Use the standard libraries in python for finding linear regression.

2.Set variables for assigning dataset values.

3.Import linear regression from sklearn.

4.Predict the values of array.

5.Calculate the accuracy, confusion and classification report b importing the required modules from sklearn.

6.Obtain the graph.

## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: SWETHA.S
RegisterNumber: 212222230155
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
data=np.loadtxt("/content/ex2data1.txt",delimiter=',')
X=data[:, [0, 1]]
y=data[:, 2]
X[:5]
y[:5]
plt.figure()
plt.scatter(X[y == 1][:, 0], X[y ==1][:, 1], label="Admitted")
plt.scatter(X[y == 0][:, 0], X[y ==0][:, 1], label=" Not Admitted")
plt.xlabel("Exam 1 score")
plt.ylabel("Exam 2 score")
plt.legend()
plt.show()
def sigmoid(z):
  return 1 / (1 + np.exp(-z))
plt.plot()
X_plot = np.linspace(-10,10,100)
plt.plot(X_plot, sigmoid(X_plot))
plt.show()
def costFunction(theta,X,y):
  h=sigmoid(np.dot(X,theta))
  j=-(np.dot(y,np.log(h))+np.dot(1-y,np.log(1-h)))/X.shape[0]
  grad=np.dot(x.T,h-y)/x.shape[0]
  return j,grad
X_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([0,0,0])
j,grad=costFunction(theta,X_train,y)
print(j)
print(grad)
x_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([-24,0.2,0.2])
j,grad=costFunction(theta,X_train,y)
print(j)
print(grad)
def cost(theta,X,y):
  h=sigmoid(np.dot(X,theta))
  j=-(np.dot(y,np.log(h))+np.dot(1-y,np.log(1-h)))/X.shape[0]
  return j
def gradient(theta,X,y):
  h=sigmoid(np.dot(X,theta))
  grad=np.dot(X.T,h-y)/X.shape[0]
  return grad
X_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([0,0,0])
res=optimize.minimize(fun=cost,x0=theta,args=(X_train,y),method='Newton-CG',jac=gradient)
print(res.fun)
print(res.x)
def plotDecisionBoundary(theta,X,y):
  x_min,x_max=X[:,0].min()-1,X[:,0].max()+1
  y_min,y_max=X[:,1].min()-1,X[:,1].max()+1
  xx,yy=np.meshgrid(np.arange(x_min,x_max,0.1),np.arange(y_min,y_max,0.1))
  X_plot=np.c_[xx.ravel(),yy.ravel()]
  X_plot=np.hstack((np.ones((X_plot.shape[0],1)),X_plot))
  y_plot=np.dot(X_plot,theta).reshape(xx.shape)
  plt.figure()
  plt.scatter(X[y == 1][:, 0], X[y ==1][:, 1], label="Admitted")
  plt.scatter(X[y == 0][:, 0], X[y ==0][:, 1], label=" Not Admitted")
  plt.contour(xx,yy,y_plot,levels=[0])
  plt.xlabel("Exam 1 score")
  plt.ylabel("Exam 2 score")
  plt.legend()
  plt.show()
prob=sigmoid(np.dot(np.array([1,45,85]),res.x))
print(prob)
def predict(theta, X):
  X_train=np.hstack((np.ones((X.shape[0],1)),X))
  prob=sigmoid(np.dot(X_train,theta))
  return (prob >= 0.5).astype(int)
np.mean(predict(res.x,X)==y)
*/
```

## Output:

Array value of X:

![image](https://github.com/swethaselvarajm/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119525603/ec0f738f-692f-48bb-8b66-2eb35f6ad5c9)

Array value of Y:

![image](https://github.com/swethaselvarajm/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119525603/c517cd69-21d7-41bb-bf06-0fd7e29aa5de)

Sigmoid function graph:

![image](https://github.com/swethaselvarajm/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119525603/932aafec-5e0e-4748-9cc7-acc66736b349)

X_Train_grad value:

![image](https://github.com/swethaselvarajm/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119525603/7e7c5659-0054-4813-8689-f5512e9912f7)

Y_Train_grad value:

![image](https://github.com/swethaselvarajm/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119525603/382739a1-957a-4b38-b40e-6ad77634d2f0)

res.X:

![image](https://github.com/swethaselvarajm/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119525603/df09a65d-3a28-4638-adfa-e6493bf553dc)

Decision boundary graph for exam score:

![image](https://github.com/swethaselvarajm/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119525603/2635371d-699b-4761-a8f7-7d497e03ab99)

probability value:

![image](https://github.com/swethaselvarajm/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119525603/05ee2c1a-7d7c-4d76-85fd-1144de32ec36)

Prediction value of mean:

![image](https://github.com/swethaselvarajm/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119525603/5a52d084-aa6e-48b2-8610-ab187bda74fd)

## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

