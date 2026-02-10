import numpy as np
import matplotlib.pyplot as plt
import pandas as pd



def loss_func(w, b, X, Y ):
    mse = 0
    for x,y in zip(X, Y):
        mse= mse + (y - (w*x+b))**2
    mse = mse/ float(len(X))
    return mse
def gradient_descent(w1, b1, X, Y, learning_rate):
    mse_w, mse_b = 0, 0
    for x,y in zip(X,Y):
        mse_w = mse_w+ -2* x * (y - (w1*x + b1))
        mse_b  = mse_b + -2* (y - (w1*x + b1)) 
    mse_w= mse_w/(float(len(X)))
    mse_b= mse_b/(float(len(X)))
    slope= w1- learning_rate*mse_w
    intercept = b1- learning_rate*mse_b
    return slope, intercept

df = pd.read_csv(r"C:\Users\Dell\task1\regression\weatherHistory.csv")
print(df.columns)
X = df["Humidity"].values
Y = df["Temperature (C)"].values
X = X[:2000]
Y = Y[:2000]
X = (X - X.mean()) / X.std()

w=0
b=0
learning_rate = 0.0001
epochs = 10000
for i in range (epochs):
     w, b = gradient_descent(w, b, X, Y, learning_rate)

print(w,b)
plt.scatter(X, Y, label='Observed Value')
plt.plot(X, w*X + b, label='Predicted Value', color='red')
plt.xlabel('<--X-Axis-->')
plt.ylabel('<--Y-Axis-->')
plt.legend()
plt.show()

