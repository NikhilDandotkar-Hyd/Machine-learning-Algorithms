import numpy as np
import pandas as pd
from random import uniform
###############
def distance_m(n1,n2):
    d=np.sum(np.square(n1-n2))
    return d
#################
data = pd.read_csv(r'reducingcostmodelsnormalized.csv')
m_data=np.matrix([data["Fixed Costs ratio"],data["equated margins"],data["Process costs"],data["Utilization rate"],data["Demand_growth"],data["Revenue"],data["RoboticAssy"],data["Rawmatl Costs"]])
m_data=m_data.transpose()
ds=m_data.shape
print (ds[0])
Clusters = input('Enter the Number of Clusters :')
threshold = input('Enter the stopping Threshold :')
a=np.matrix(np.random.uniform(-3,1,size=(Clusters,ds[1])))
#print (a)
#print (m_data[0,:])
dd=np.zeros((ds[0],Clusters))
err = threshold+1
while (err>threshold):
    # Finding distances from centroid
    for y in range(0, ds[0]):
        for x in range(0, Clusters):
            dd[y, x] = distance_m(a[x, :], m_data[y, :])
    k_clusters = np.zeros(ds[0])
    count = np.zeros(Clusters)
    k_center = np.zeros((Clusters, ds[1]))
    # Assigning Clusters
    for i in range(0, ds[0]):
        k_clusters[i] = np.argmin(dd[i, :])
    # print (k_clusters)
    # Finding cluster centroid
    for jj in range(0, Clusters):
        for ii in range(0, ds[0]):
            if k_clusters[ii] == jj:
                count[jj] = count[jj] + 1
                k_center[jj, :] = k_center[jj, :] + m_data[ii, :]
    for jj in range(0, Clusters):
        k_center[jj, :] = k_center[jj, :] / count[jj]
    # print (k_center)
    dc=0
    for jj in range(0, Clusters):
        dc = dc + distance_m(a[jj, :], k_center[jj, :])
    err = np.sqrt(dc)
    print(err)
    a = k_center