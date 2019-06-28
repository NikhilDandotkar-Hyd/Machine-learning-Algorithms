import numpy as np

from glob import glob

# loading the traning data
feature_filename = glob('knn-dataset_train\data*.csv')
label_filename = glob('knn-dataset_train\labels*.csv')
feature_array = np.genfromtxt('knn-dataset_train\data1.csv', delimiter=',')
label_array = np.genfromtxt('knn-dataset_train\labels1.csv', delimiter=',')

for f in feature_filename[1:]:
    temp = np.genfromtxt(f, delimiter=',')
    feature_array = np.concatenate((feature_array, temp), axis=0)
for l in label_filename[1:]:
    temp1 = np.genfromtxt(l, delimiter=',')
    label_array = np.concatenate((label_array, temp1), axis=0)

# loading the testing data
test_feature_array = np.genfromtxt('data10.csv', delimiter=',')
test_label_array = np.genfromtxt('labels10.csv', delimiter=',')

# Testing point
test_point = np.array(test_feature_array[1, :])

# Calculating the distance between the test point and training data
no_row, no_col = feature_array.shape
temp_data = np.repeat(test_point, no_row, axis=0)
temp_data = np.reshape(temp_data,(no_row,no_col))

squ_diff = np.square(feature_array - temp_data)

dis = np.square(np.sum(squ_diff, axis=1))
rank_dis = np.argsort(dis)

# Taking the 'K' for comparision
K_value = int(input('K value : '))

# considering 'K' nearest neighbor's distances
votes = []
for i in range(rank_dis.size):
    if rank_dis[i] < K_value:
         votes.append(label_array[i])

# Assigning the class
winner = np.median(votes)
print("predicted class :",winner)
print(("actual class ", test_label_array[1]))




