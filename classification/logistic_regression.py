import numpy as np
import pandas as pd

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))
def bce(Y, Y_HAT):#binary cross entropy
    loss=0
    eps= 1e-9
    for y, y_hat in zip(Y, Y_HAT):
        loss= loss + -1.0*(y*np.log(y_hat+eps)+(1-y)*np.log(1-y_hat+eps))
    return loss/float(len(Y))
def gradient(w, b, X,Y, alpha):
    dw= np.zeros_like(w)
    db=0
    for x, y in zip(X,Y):
        z=np.dot(w,x)+b
        y_hat= sigmoid(z)
        dw= dw+ (y_hat-y)*x
        db= db+ (y_hat-y)
    dw= dw/len(X)
    db= db/len(X)

    w= w-alpha*dw
    b= b- alpha*db
    return w, b
def probability(X, w, b):
    prob=[]
    for x in X:
        prob.append(sigmoid(np.dot(w,x)+b))
    return np.array(prob)
def predict(X, w, b, threshold=0.5):
    y_pred = []
    probs = probability(X, w, b)

    for p in probs:
        if p >= threshold:
            y_pred.append(1)
        else:
            y_pred.append(0)

    return np.array(y_pred)

def accuracy(y_true, y_pred):
    correct = 0
    for y, y_hat in zip(y_true, y_pred):
        if y == y_hat:
            correct += 1
    return correct / len(y_true)

df = pd.read_csv(r"C:\Users\Dell\Desktop\reaggression\classification\candy-data.csv")

#probability that a candy is popular
median_win = df["winpercent"].median()
df["label"] = (df["winpercent"] > median_win).astype(int)


features = [
    "chocolate",
    "fruity",
    "caramel",
    "peanutyalmondy",
    "nougat",
    "crispedricewafer"
]

X = df[features].values
y = df["label"].values


w = np.zeros(X.shape[1])
b = 0

learning_rate = 0.05
epochs = 2000


for epoch in range(epochs):
    w, b = gradient(w, b, X, y, learning_rate)

    if epoch % 200 == 0:
        y_pred = predict(X, w, b)
        acc = accuracy(y, y_pred)
        print(f"Epoch {epoch:4d} | Accuracy: {acc:.3f}")


y_pred = predict(X, w, b)
print("Final Accuracy:", accuracy(y, y_pred))
import matplotlib.pyplot as plt


probs = probability(X, w, b)

plt.figure(figsize=(8,5))
plt.scatter(range(len(probs)), probs, c=y, cmap="bwr", alpha=0.6)
plt.axhline(0.5, color='black', linestyle='--', label="Decision Threshold")
plt.xlabel("Sample Index")
plt.ylabel("Predicted Probability (Class = 1)")
plt.title("Logistic Regression Predictions")
plt.legend()
plt.show()
