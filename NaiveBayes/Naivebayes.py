import numpy as np
import math

############### Gaussian distribution
def p_gaus(x, mean, sigma):
    PI = math.pi
    x_p = -0.5*math.pow((x - mean),2)/sigma
    p = (1 / (math.sqrt(2 * PI * sigma))) * math.pow(math.e,x_p )
    return p



y_matrix= np.genfromtxt('Y.csv', delimiter=',')
t_matrix= np.genfromtxt('t.csv', delimiter=',')
m=t_matrix.shape
class1=[]
class2=[]
for i in range(0,m[0]):
    if  t_matrix[i] == 1:
        class1 = np.concatenate((class1, y_matrix[:,i]),axis=0)
    else:
        class2 = np.concatenate((class2, y_matrix[:, i]))
c1=np.reshape(class1, (int(class1.size/2), 2))
c2=np.reshape(class2, (int(class2.size/2), 2))
mean0_c1, mean1_c1 = np.mean(c1, axis=0)
mean0_c2, mean1_c2 = np.mean(c2, axis=0)
var0_c1, var1_c1 = np.var(c1, axis=0)
var0_c2, var1_c2 = np.var(c2, axis=0)

############### test vectors
yn=np.matrix('1,1,-1,-1;1,-1,1,-1')

for i in range(0,4):
    if p_gaus(yn[0,i],mean0_c1,var0_c1)*p_gaus(yn[1,i],mean1_c1,var1_c1) > p_gaus(yn[0,i],mean0_c2,var0_c2)*p_gaus(yn[1,i],mean1_c2,var1_c2):
        print(yn[:,i],'belongs to class 1')
    else:
        print(yn[:,i],'belongs to class -1')