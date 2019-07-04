from numpy import *
import numpy

def mean_var(data):
    N=data.shape
    m = float(sum(data)) / N[1]
    v = float(sum(square(data - m))) / N[1]
    return m,v
n=10
p=0.2
N=100000
data1=numpy.random.binomial(n,p,size=[1,N])
m_e,v_e=mean_var(data1)
p_e=1-float(v_e/m_e)
n_e=float(m_e/p_e)
ext_data1=numpy.random.binomial(n_e,p_e,size=[1,N])
err=numpy.mean(numpy.subtract(data1,ext_data1)**2)
print('Mean Squared Error in Binomial Estimation:%.3f'%(err))
#############################################################
lam=1
data2=numpy.random.poisson(lam,size=[1,N])
m_e,v_e=mean_var(data2)
ext_data2=numpy.random.poisson(m_e,size=[1,N])
err=numpy.mean(numpy.subtract(data2,ext_data2)**2)
print('Mean Squared Error in poisson Estimation:%.3f'%(err))
#############################################################
lam=2
data3=numpy.random.exponential(lam,size=[1,N])
m_e,v_e=mean_var(data3)
lam_e=1/m_e
ext_data3=numpy.random.exponential(lam,size=[1,N])
err=numpy.mean(numpy.subtract(data3,ext_data3)**2)
print('Mean Squared Error in exponential Estimation:%.3f'%(err))
##############################################################
me=0
va=1
data4=numpy.random.normal(me,va,size=[1,N])
m_e,v_e=mean_var(data4)
ext_data4=numpy.random.normal(m_e,v_e,size=[1,N])
err=numpy.mean(numpy.subtract(data4,ext_data4)**2)
print('Mean Squared Error in Gaussian Estimation:%.3f'%(err))
###############################################################
me=0
va=1
data5=numpy.random.laplace(me,va,size=[1,N])
m_e,v_e=mean_var(data5)
ext_data5=numpy.random.normal(m_e,v_e,size=[1,N])
err=numpy.mean(numpy.subtract(data5,ext_data5)**2)
print('Mean Squared Error in Laplacian Estimation:%.3f'%(err))