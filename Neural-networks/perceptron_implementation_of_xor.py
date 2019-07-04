import numpy as np
from math import exp


# activation function
def activation(x):
    y = 1/(1+exp(-x))
    return y


# gradient of activation function
def grad_activation(x):
    gy = activation(x)*(1-activation(x))
    return gy


# weights
w11, w12, w21, w22, w13, w23 = 20, -20, 20, -20, 30, 30
pw11,pw12,pw21,pw22,pw13,pw23=w11,w12,w21,w22,w13,w23
# bias
b1, b2, b3 = -10, 30, -30
pb1, pb2, pb3 = b1, b2, b3
# inputs
x=np.matrix('0,0,1,1;0,1,0,1')
d=np.matrix('0,1,1,0')
alpha=0.0001
eta=0.25
y3=np.matrix('5,5,5,5')

# before trainng
print ('before training ')

for i in range(0, 4):
    x1, x2 = x[0, i], x[1, i]
    # input to 1st hidden neuron
    v1 = w11 * x1 + w21 * x2 + b1
    # input to 2nd hidden neuron
    v2 = w12 * x1 + w22 * x2 + b2
    # output from 1st hidden neuron
    y1 = activation(v1)
    # output from 2nd hidden neuron
    y2 = activation(v2)
    # input to 1st output neuron
    v3 = w13 * y1 + w23 * y2 + b3
    # output from 1st output neuron
    y3[0,i] = activation(v3)
    print ('output for',x[:,i],'is',y3[0,i])
y3=np.matrix('5,5,5,5')
# training
print ('training')

for i in range(0,4):
    while y3[0,i] != d[0,i]:
        x1, x2 = x[0, i], x[1, i]
        # input to 1st hidden neuron
        v1 = w11 * x1 + w21 * x2 + b1
        # input to 2nd hidden neuron
        v2 = w12 * x1 + w22 * x2 + b2
        # output from 1st hidden neuron
        y1 = activation(v1)
        # output from 2nd hidden neuron
        y2 = activation(v2)
        # input to 1st output neuron
        v3 = w13 * y1 + w23 * y2 + b3
        # output from 1st output neuron
        y3[0,i] = activation(v3)
        if (d[0, i] - y3[0,i])==0:
            continue
        print ('error is',d[0,i]-y3[0,i])
        # gradient equation
        delta_o = grad_activation(v3) * (d[0, i] - y3[0,i])
        delta_h1 = grad_activation(v1) * delta_o * w13
        delta_h2 = grad_activation(v2) * delta_o * w23
        ########################
        tw11, tw12, tw21, tw22, tw13, tw23 = w11, w12, w21, w22, w13, w23
        tb1, tb2, tb3 = b1, b2, b3
        # backpropagation ####
        w13 = w13 + alpha * pw13 + eta * delta_o * y1
        w23 = w23 + alpha * pw23 + eta * delta_o * y2
        w11 = w11 + alpha * pw11 + eta * delta_h1 * x1
        w12 = w12 + alpha * pw12 + eta * delta_h2 * x1
        w21 = w21 + alpha * pw21 + eta * delta_h1 * x2
        w22 = w22 + alpha * pw22 + eta * delta_h2 * x2
        b1 = b1 + alpha * pb1 + eta * delta_h1
        b2 = b2 + alpha * pb2 + eta * delta_h2
        b3 = b3 + alpha * pb3 + eta * delta_o
        pw11, pw12, pw21, pw22, pw13, pw23 = tw11, tw12, tw21, tw22, tw13, tw23
        pb1, pb2, pb3 = tb1, tb2, tb3
# after training
print ('after training ')

for i in range(0, 4):
    x1, x2 = x[0, i], x[1, i]
    # input to 1st hidden neuron
    v1 = w11 * x1 + w21 * x2 + b1
    # input to 2nd hidden neuron
    v2 = w12 * x1 + w22 * x2 + b2
    # output from 1st hidden neuron
    y1 = activation(v1)
    # output from 2nd hidden neuron
    y2 = activation(v2)
    # input to 1st output neuron
    v3 = w13 * y1 + w23 * y2 + b3
    # output from 1st output neuron
    y3[0,i] = activation(v3)
    print ('output for',x[:,i],'is',y3[0,i])
