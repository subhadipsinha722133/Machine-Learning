import numpy as np

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 1])
print(X)
print(y)

w = np.array([1, 1])
print(w)

b = -0.5
pred = []
for a in X:
    y_hat = np.dot(a, w) + b
    # pred.append(activation(y_hat))

print(pred)


def activation(z):
    if z >= 0:
        return 1
    else:
        return 0


# Perceptron Learning

import math
import numpy as np

epochs = 100
alpha = 0.2

w0 = np.random.random()
w1 = np.random.random()
w2 = np.random.random()
print("initial Weights:")
print("w0=", w0, "w1=", w1, "w2=", w2)

del_w0 = 1
del_w1 = 1
del_w2 = 1
train_data_temp = [
    [0, 0, 0],
    [0, 0, 1],
    [0, 1, 0],
    [0, 1, 1],
    [1, 0, 0],
    [1, 0, 1],
    [1, 1, 0],
    [1, 1, 1],
]
train_data = np.asarray(train_data_temp)
op = np.array([0, 1, 1, 1, 1, 1, 1, 1])

print(train_data)
print(op)

# y=(x*w)+b

bias = 0
for i in range(epochs):
    j = 0
    for x in train_data:
        y_hat = w0 * x[0] + w1 * x[1] + w2 * x[2] + bias

        if y_hat >= 0:
            act = 1
        else:
            act = 0

        err = op[j] - act

        del_w0 = alpha * x[0] * err
        del_w1 = alpha * x[1] * err
        del_w2 = alpha * x[2] * err

        W0 = w0 + del_w0
        w1 = w1 + del_w1
        w2 = w2 + del_w2

        j = j + 1
        print("epoch ", i + 1, "error = ", err)
        print(del_w0, del_w1, del_w2)

print("\nFinal Weights =")
print("w0=", w0, "w1=", w1, "w2=", w2)
