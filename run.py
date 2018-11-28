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
import data_util
from tfrbm.cgbrbm import CGBRBM
import matplotlib.pyplot as plt
# from sklearn.manifold import TSNE
from MulticoreTSNE import MulticoreTSNE as TSNE
import pickle

# define
base_data_dir = '../Data/Image_Classification'


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


print("loading training and validation data...")
start_time = time.time()
cgbrbm = CGBRBM(n_visible=8192, n_hidden=800, chanel_increase=16)
x_train, y_train = data_util.create_raw_data(base_data_dir)
time_dif = get_time_dif(start_time)
print("Time usage:", time_dif)

#
print('Training and evaluating...')
start_time = time.time()
errs = cgbrbm.cfit(x_train, y_train, n_epoches=20, batch_size=30)
time_dif = get_time_dif(start_time)
print("Time usage:", time_dif)

print('Print error plot')
start_time = time.time()
plt.plot(errs)
plt.show()
time_dif = get_time_dif(start_time)
print("Time usage:", time_dif)

print('Getting hidden layers...')
start_time = time.time()
hidden_layers = np.zeros((3000, 800))
hidden_layer_counter = 0
for file in os.listdir('./pickle')[0:100]:
    with open('./pickle/' + file, 'rb') as p:
        temp = pickle.load(p)
        hidden_layers_batch = cgbrbm.ctransform(temp, batch_size=30)  # 得到hidden layer
        for i in range(len(hidden_layers_batch)):
            hidden_layers[hidden_layer_counter] = hidden_layers_batch[i]
            hidden_layer_counter += 1
time_dif = get_time_dif(start_time)
print("Time usage:", time_dif)

print('Getting labels...')
start_time = time.time()
label_list = []
for file in os.listdir('./pickle')[100:200]:
    with open('./pickle/' + file, 'rb') as p:
        temp = pickle.load(p)
        label_list.extend([temp[i].decode("utf-8") for i in range(30)])
time_dif = get_time_dif(start_time)
print("Time usage:", time_dif)

print('TSNE...')
start_time = time.time()
embedded_dim = 2
tsne = TSNE(n_components=embedded_dim, random_state=0, n_jobs=4)
np.set_printoptions(suppress=True)
new_dim_points = tsne.fit_transform(hidden_layers)
time_dif = get_time_dif(start_time)
print("Time usage:", time_dif)

print('get points seperated...')
start_time = time.time()
point_list_16 = []
for i in range(len(label_list)):
    if label_list[i][29] == '1':
        point_list_16.append(new_dim_points[i])
point_list_64 = []
for i in range(len(label_list)):
    if label_list[i][29] == '6':
        point_list_64.append(new_dim_points[i])
point_list_Q = []
for i in range(len(label_list)):
    if label_list[i][29] == 'Q':
        point_list_Q.append(new_dim_points[i])
time_dif = get_time_dif(start_time)
print("Time usage:", time_dif)

print('Plot TSNE...')
start_time = time.time()
fig, ax = plt.subplots()
print_tsne(data=point_list_16, label='16', color='dodgerblue', axis=ax)
print_tsne(data=point_list_64, label='64', color='red', axis=ax)
print_tsne(data=point_list_Q, label='Q', color='gold', axis=ax)
plt.show()
time_dif = get_time_dif(start_time)
print("Time usage:", time_dif)

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
