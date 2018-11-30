#
# -*- coding: utf-8 -*-

# import
import os
import sys
import time
from datetime import timedelta
import numpy as np
import tensorflow as tf
from sklearn import metrics
from sklearn.cluster import DBSCAN
import data_util
from tfrbm.cgbrbm import CGBRBM
import matplotlib.pyplot as plt
# from sklearn.manifold import TSNE
from MulticoreTSNE import MulticoreTSNE as TSNE
import pickle

# define functions
def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


def print_tsne(data, label, color, axis):
    for _ in data:
        c = color
        m = "^"
        axis.scatter(_[0], _[1], c=c, marker=m, label=label)
    print(label, 'finished')
    
#%% define variables
base_data_dir = '../Data/Image_Classification'

chanel_increase=16
filter_height=5  # 卷积核尺寸
filter_width=5  # 卷积核尺寸
offset_height=80
offset_width=270
target_height=700
target_width=700
resize_height=64
resize_width=64
resize_method=0
num_each_cat=1000
total_cat = None
total_spy = 6
total_input = None

spy_batch_size = 6
batch_size = 30
n_visible=(resize_height / 4 * resize_width / 4) * (chanel_increase * 2)
n_hidden=1000

#%% loading data & initilize cgbrbm...
print("loading data & initilize cgbrbm...")
start_time = time.time()
cgbrbm = CGBRBM(int(n_visible), 
                n_hidden, 
                chanel_increase,
                filter_height,  # 卷积核尺寸
                filter_width,  # 卷积核尺寸
                offset_height,
                offset_width,
                target_height,
                target_width,
                resize_height,
                resize_width,
                resize_method)

x_train, y_train, total_cat = data_util.create_raw_data(base_data_dir, num_each_cat)
x_spy, y_spy = data_util.create_spy_data(base_data_dir, total_spy)

time_dif = get_time_dif(start_time)
print("Time usage:", time_dif)

#%% Training and evaluating...
print('Training and evaluating...')
start_time = time.time()
errs = cgbrbm.cfit(x_train, y_train, n_epoches=10, batch_size=batch_size)
time_dif = get_time_dif(start_time)
print("Time usage:", time_dif)

#%% Print error plot
print('Print error plot')
start_time = time.time()
plt.plot(errs)
plt.show()
time_dif = get_time_dif(start_time)
print("Time usage:", time_dif)

#%% Getting hidden layers & labels...
print('Getting hidden layers & labels...')
start_time = time.time()
total_input = num_each_cat * total_cat + total_spy
hidden_layers = np.zeros((total_input, n_hidden))
hidden_layer_counter = 0
for file in os.listdir('./pickle')[0:100]:
    with open('./pickle/' + file, 'rb') as p:
        temp = pickle.load(p)
        hidden_layers_batch = cgbrbm.ctransform(temp, batch_size=batch_size)  # 得到hidden layer
        for i in range(len(hidden_layers_batch)):
            hidden_layers[hidden_layer_counter] = hidden_layers_batch[i]
            hidden_layer_counter += 1
            
#%%
label_list = []
for file in os.listdir('./pickle')[100:200]:
    with open('./pickle/' + file, 'rb') as p:
        temp = pickle.load(p)
        label_list.extend([temp[i].decode("utf-8") for i in range(batch_size)])
        
#%%
spy_hidden_layers = cgbrbm.ctransform_raw(x_spy, y_spy, spy_batch_size)

for i in range(len(spy_hidden_layers)):
    hidden_layers[hidden_layer_counter] = spy_hidden_layers[i]
    hidden_layer_counter += 1


#%%
for file in os.listdir('./pickle')[200 + int(total_spy / spy_batch_size):]:
    with open('./pickle/' + file, 'rb') as p:
        temp = pickle.load(p)
        label_list.extend([temp[i].decode("utf-8") for i in range(spy_batch_size)])
time_dif = get_time_dif(start_time)
print("Time usage:", time_dif)

#%% TSNE
print('TSNE...')
start_time = time.time()
embedded_dim = 2
tsne = TSNE(n_components=embedded_dim, random_state=0, n_jobs=4)
np.set_printoptions(suppress=True)
new_dim_points = tsne.fit_transform(hidden_layers)
time_dif = get_time_dif(start_time)
print("Time usage:", time_dif)

#%%
print('get points seperated by label...')
start_time = time.time()
label_16 = []
for i in range(len(label_list)):
    if label_list[i][29] == '1':
        label_16.append(np.append(np.copy(new_dim_points[i]), [i]))
label_16.append(np.append(np.copy(new_dim_points[-6]), [total_input - 6]))
label_16.append(np.append(np.copy(new_dim_points[-5]), [total_input - 5]))

label_64 = []
for i in range(len(label_list)):
    if label_list[i][29] == '6':
        label_64.append(np.append(np.copy(new_dim_points[i]), [i]))
label_64.append(np.append(np.copy(new_dim_points[-4]), [total_input - 4]))
label_64.append(np.append(np.copy(new_dim_points[-3]), [total_input - 3]))
    
label_Q = []
for i in range(len(label_list)):
    if label_list[i][29] == 'Q':
        label_Q.append(np.append(np.copy(new_dim_points[i]), [i]))
label_Q.append(np.append(np.copy(new_dim_points[-2]), [total_input - 2]))
label_Q.append(np.append(np.copy(new_dim_points[-1]), [total_input - 1]))
        
label_16 = np.array(label_16)
label_64 = np.array(label_64)
label_Q = np.array(label_Q)
time_dif = get_time_dif(start_time)
print("Time usage:", time_dif)

#%%
print('DBSCAN...')
start_time = time.time()
dbscan_label = DBSCAN(eps=2, min_samples=4).fit_predict(new_dim_points)
plt.scatter(new_dim_points[:, 0], new_dim_points[:, 1], c=dbscan_label)
plt.show()
time_dif = get_time_dif(start_time)
print("Time usage:", time_dif)

#%%
print('get points seperated by dbscan...')
assert dbscan_label[-6] == dbscan_label[-5]
assert dbscan_label[-4] == dbscan_label[-3]
assert dbscan_label[-2] == dbscan_label[-1]
start_time = time.time()
dbscan_16 = []
dbscan_64 = []
dbscan_Q = []
counter = 0
for j in range(total_cat):
    assert j < total_cat
    temp_set = []
    counter = 0
    for i in range(len(dbscan_label)):
        if dbscan_label[i] == j:
            temp_set.append(np.append(np.copy(new_dim_points[i]), [i]))
            counter += 1
    print(counter)
    assert counter == total_input/total_cat
    if dbscan_label[-6] == j:
        dbscan_16 = np.array(temp_set.copy())
    if dbscan_label[-4] == j:
        dbscan_64 = np.array(temp_set.copy())
    if dbscan_label[-2] == j:
        dbscan_Q = np.array(temp_set.copy())
time_dif = get_time_dif(start_time)
print("Time usage:", time_dif)

#%% compute accuracy
incorrect16 = max(len(np.setdiff1d(dbscan_16[:, 2], label_16[:, 2])),
              len(np.setdiff1d(label_16[:, 2], dbscan_16[:, 2]))) 
incorrect64 = max(len(np.setdiff1d(dbscan_64[:, 2], label_64[:, 2])), 
              len(np.setdiff1d(label_64[:, 2], dbscan_64[:, 2])))
incorrectQ = max(len(np.setdiff1d(dbscan_Q[:, 2], label_Q[:, 2])), 
              len(np.setdiff1d(label_Q[:, 2], dbscan_Q[:, 2])))
correct = total_input - total_spy - incorrect16 - incorrect64 - incorrectQ
accuracy = correct / (total_input - total_spy)
print(accuracy)
#%%
print('Plot TSNE by label...')
start_time = time.time()
fig, ax = plt.subplots()
print_tsne(data=label_16, label='16', color='dodgerblue', axis=ax)
print_tsne(data=label_64, label='64', color='red', axis=ax)
print_tsne(data=label_Q, label='Q', color='gold', axis=ax)
plt.show()
time_dif = get_time_dif(start_time)
print("Time usage:", time_dif)

#%%
print('Plot TSNE by dbscan...')
start_time = time.time()
fig, ax = plt.subplots()
print_tsne(data=dbscan_16, label='16', color='dodgerblue', axis=ax)
print_tsne(data=dbscan_64, label='64', color='red', axis=ax)
print_tsne(data=dbscan_Q, label='Q', color='gold', axis=ax)
plt.show()
time_dif = get_time_dif(start_time)
print("Time usage:", time_dif)


#%% useless
#for i in point_list_64:
#    if i[0] < 0 and i[1] >15:
#        print(i[2])




##test temp
# print('Getting test errors...')
# start_time = time.time()
# errs_test = cgbrbm.cget_error(x_test, 30)
# errs_test_avg = np.average(errs_test)
# time_dif = get_time_dif(start_time)
# print("Time usage:", time_dif)
#
##t1
# print('Getting hidden layers...')
# start_time = time.time()
# hidden_layers_test = cgbrbm.ctransform(x_test, batch_size=30)  # 得到hidden layer
# time_dif = get_time_dif(start_time)
# print("Time usage:", time_dif)
#
# print('TSNE...')
# start_time = time.time()
# new_dim_points_test = tsne.fit_transform(hidden_layers_test)
# time_dif = get_time_dif(start_time)
# print("Time usage:", time_dif)
#
# print('Plot TSNE...')
# start_time = time.time()
# fig, ax = plt.subplots()
# print_tsne(data=new_dim_points_test[0:1000], label='sports', color='firebrick', axis=ax)
# print_tsne(data=new_dim_points_test[1000:2000], label='entertainment', color='red', axis=ax)
# print_tsne(data=new_dim_points_test[20000:3000], label='furniture', color='coral', axis=ax)
# print_tsne(data=new_dim_points_test[3000:4000], label='housing', color='chocolate', axis=ax)
# print_tsne(data=new_dim_points_test[4000:5000], label='education', color='darkorange', axis=ax)
# print_tsne(data=new_dim_points_test[5000:6000], label='fashion', color='gold', axis=ax)
# print_tsne(data=new_dim_points_test[6000:7000], label='politics', color='olive', axis=ax)
# print_tsne(data=new_dim_points_test[7000:8000], label='gaming', color='yellow', axis=ax)
# print_tsne(data=new_dim_points_test[8000:9000], label='social', color='lawngreen', axis=ax)
# print_tsne(data=new_dim_points_test[9000:10000], label='tech', color='mediumturquoise', axis=ax)
# print_tsne(data=new_dim_points_test[10000:11000], label='economy', color='dodgerblue', axis=ax)
##ax.legend()
# plt.show()
# time_dif = get_time_dif(start_time)
# print("Time usage:", time_dif)


