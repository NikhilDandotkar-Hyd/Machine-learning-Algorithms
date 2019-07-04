from cvxpy import *
import numpy

def predict(w0,w1,b,x1,x2):
    return numpy.sign(w0*x1+w1*x2+b)
############ input data
t = numpy.genfromtxt('t.csv', delimiter=',')
X = numpy.genfromtxt('X.csv', delimiter=',')
############ optimization
w0=Variable()
w1=Variable()
b=Variable()
constraint=[]
for i in range(len(t)):
    if t[i]==1:
     constraint=[w0*X[i,0]+w1*X[i,1]+b>=1]+constraint
    else:
     constraint=[w0*X[i,0]+w1*X[i,1]+b<=-1]+constraint

obj = Minimize(0.5*(square(w0)+square(w1)))

prob = Problem(obj,constraint)
prob.solve()
########### test data
test=numpy.array([[2,0.5],[-0.8,-0.7],[1.58,-1.33],[-0.008,0.001]])
########### prediction on test data
for i in range(0,4):
    print ("point",test[i,:],"belongs to",predict(w0.value,w1.value,b.value,test[i,0],test[i,1]),"class")