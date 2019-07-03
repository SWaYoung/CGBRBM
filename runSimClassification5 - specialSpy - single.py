#
# -*- coding: utf-8 -*-

# import
import os
import time
from datetime import timedelta
import numpy as np
# import tensorflow as tf
from sklearn.cluster import DBSCAN
import data_util
from tfrbm.cgbrbm import CGBRBM
import matplotlib.pyplot as plt
# from sklearn.manifold import TSNE
from MulticoreTSNE import MulticoreTSNE as TSNE
import pickle
import math

# 总体时间
total_start_time = time.time()


# define functions
def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


def print_tsne(data, label, color):
    for _ in data:
        c = color
        m = "^"
        plt.scatter(_[0], _[1], c=c, marker=m, label=label)
    print(label, 'finished')

def Nminelements(list1, N): 
    final_list = []  
    for i in range(0, N):  
        min1 = max(list1)
        for j in range(len(list1)):      
            if list1[j] < min1: 
                min1 = list1[j]                  
        list1.remove(min1)
        final_list.append(min1) 
    return(final_list)

def get_centroid(dbscan):
    x_coords = [p[0] for p in dbscan]
    y_coords = [p[1] for p in dbscan]
    _len = len(dbscan)
    centroid_x = sum(x_coords)/_len
    centroid_y = sum(y_coords)/_len
    return [centroid_x, centroid_y]

def calculate_distance_centroid(dbscan, spy_list):
    result_list = []
    for i in range(len(spy_list)):
        result_list.append(math.sqrt(((dbscan[0] - spy_list[i][0])**2) + ((dbscan[1] - spy_list[i][1])**2)))
    return result_list.index(min(result_list))

def calculate_distance_centroid_v2(dbscan, dbscan_list):
    result_list = []
    for i in range(len(dbscan_list)):
        result_list.append(math.sqrt(((dbscan[0] - dbscan_list[i][0])**2) + ((dbscan[1] - dbscan_list[i][1])**2)))
    result_list = list(filter(lambda x: x>0, result_list))
    result_list = Nminelements(result_list, 1)
    return sum(result_list)/len(result_list)/2

def calculate_closest_points(dbscan, spy_list):
    result_list = []
    for i in range(len(spy_list)):
        count = 0
        for j in range(len(spy_list[i])):
            distance = math.sqrt(((dbscan[0] - spy_list[i][j][0])**2) + ((dbscan[1] - spy_list[i][j][1])**2))    
            if distance < 2.5:
                count += 1
        result_list.append(count)
    return result_list.index(max(result_list))

def calculate_closest_points_v2(dbscan, spy_list):
    result_list = []
    for i in range(len(spy_list)):
        distance_sum = 0
        for j in range(len(spy_list[i])):
            distance = math.sqrt(((dbscan[0] - spy_list[i][j][0])**2) + ((dbscan[1] - spy_list[i][j][1])**2))    
            if distance < 2.5:
                distance_sum += distance
        if distance_sum == 0:
            distance_sum = 10000
        result_list.append(distance_sum)
    return result_list.index(min(result_list))

def calculate_distance_all(dbscan, spy_list):
    result_list = []
    for i in range(len(spy_list)):
        distance_list = []
        distance_sum = 0
        for j in range(len(spy_list[i])):
            distance = math.sqrt(((dbscan[0] - spy_list[i][j][0])**2) + ((dbscan[1] - spy_list[i][j][1])**2))    
            distance_list.append(distance)  
        distance_sum = sum(distance_list)
        result_list.append(distance_sum)
    return result_list.index(min(result_list))

def calculate_percentage(dbscan, spy_list, distance):
    result_list = []
    total = 0
    for i in range(len(spy_list)):
        count = 0
        for j in range(len(spy_list[i])):
            calculated_distance = math.sqrt(((dbscan[0] - spy_list[i][j][0])**2) + ((dbscan[1] - spy_list[i][j][1])**2))    
            if calculated_distance < distance:
                count += 1
                total += 1
        result_list.append(count)
    if total!=0:
        result_list = [x/total for x in result_list]
    return result_list

# %% define variables
base_data_dir = '../Data/Image_Classification'  # 所有数据文件夹的根目录

chanel_increase = 6  # 第一层卷积后的chanel数量 16
filter_height = 3  # 卷积核尺寸 5
filter_width = 3  # 卷积核尺寸 5
offset_height = 80  # 截取图片的起始高度（从上至下）80
offset_width = 270  # 截取图片的其实宽度（从左至右）270
target_height = 350  # 截取图片的高度 350
target_width = 350  # 截取图片的宽度 350
resize_height = 64  # 缩放图片的高度 64
resize_width = 64  # 缩放图片的宽度 64
resize_method = 3  # 缩放图片的方式 0
num_each_cat = 1  # 每类文件提取多少
total_spy = 160  # spy 的总数量，目前5个类，每个类16个，每个OSNR一个
n_epochs = 1
spy_batch_size = 30  # spy的batch数量
batch_size = 30  # 输入文件的batch数量
n_visible = int(resize_height / 4 * resize_width / 4) * (chanel_increase * 2)  # 显层数量（自动计算） 8192
n_hidden = 3000  # 隐层数量 3000

# %% loading data & initilize cgbrbm...
print("loading data & initilize cgbrbm...")
start_time = time.time()
cgbrbm = CGBRBM(n_visible,  # initialize cgbrbm
                n_hidden,
                chanel_increase,
                filter_height,
                filter_width,
                offset_height,
                offset_width,
                target_height,
                target_width,
                resize_height,
                resize_width,
                resize_method)

x_train, y_train, total_cat = data_util.create_raw_data(base_data_dir, num_each_cat)  # 得到输入，和一共多少种类total_cat
x_spy, y_spy = data_util.create_spy_data(base_data_dir, total_spy)  # spy 要用平均质量的图片
time_dif = get_time_dif(start_time)
print("Time usage:", time_dif)

# %% Training and evaluating...
print('Training and evaluating...')
start_time = time.time()
errs = cgbrbm.cfit(x_train, y_train, n_epochs=n_epochs, batch_size=batch_size)  # train cgbrbm and get error
time_dif = get_time_dif(start_time)
print("Time usage:", time_dif)

# %% Print error plot
print('Print error plot')
start_time = time.time()
plt.figure()
plt.plot(errs)
plt.show()
time_dif = get_time_dif(start_time)
print("Time usage:", time_dif)

# %% Getting hidden layers & labels...
print('Getting hidden layers & labels...')
start_time = time.time()
# get inpu hidden layers
total_input = num_each_cat * total_cat + total_spy  # 得到全部数据，所有非spy输入+所有spy
hidden_layers = np.zeros((total_input, n_hidden))  # 创建numpy array，大小（所有输入，隐层数量）
hidden_layer_counter = 0
for file in os.listdir('./pickle')[0:int(math.ceil(total_cat * num_each_cat / batch_size))]:  # 所有非spy输入
    with open('./pickle/' + file, 'rb') as p:  # 从保存的pickle文件载入数据
        temp = pickle.load(p)
        hidden_layers_batch = cgbrbm.ctransform(temp, batch_size=batch_size)  # 得到hidden layer
        for i in range(len(hidden_layers_batch)):
            hidden_layers[hidden_layer_counter] = hidden_layers_batch[i]
            hidden_layer_counter += 1

# get spy hidden layers
spy_hidden_layers = cgbrbm.ctransform_spy(x_spy, y_spy, spy_batch_size)  # 把spy图片进行处理并进行卷积，然后得到隐层
for i in range(len(spy_hidden_layers)):
    hidden_layers[hidden_layer_counter] = spy_hidden_layers[i]
    hidden_layer_counter += 1

# get input labels 得到所有的输入的label,包括spy，顺序与输入数据的feature一一对应
label_list = []
start = int(math.ceil(total_cat * num_each_cat / batch_size) + math.ceil(total_spy / spy_batch_size))
end = int(start + math.ceil(total_cat * num_each_cat / batch_size))
for file in os.listdir('./pickle')[start:end]:
    with open('./pickle/' + file, 'rb') as p:
        temp = pickle.load(p)
        label_list.extend([temp[i].decode("utf-8") for i in range(len(temp))])  # 注意要decode
#        print(file)

# get spy labels
start = end
for file in os.listdir('./pickle')[start:]:
#    print('--'+file)
    with open('./pickle/' + file, 'rb') as p:
        temp = pickle.load(p)
        label_list.extend([temp[i].decode("utf-8") for i in range(len(temp))])  # 注意： 要用spy_batch_size
time_dif = get_time_dif(start_time)
print("Time usage:", time_dif)

# %% TSNE
print('TSNE...')
start_time = time.time()
embedded_dim = 2
tsne = TSNE(n_components=embedded_dim, random_state=0, n_jobs=4)
np.set_printoptions(suppress=True)
new_dim_points = tsne.fit_transform(hidden_layers)  # TSNE 对隐层进行降维，得到新的坐标点
time_dif = get_time_dif(start_time)
print("Time usage:", time_dif)

# %%对与新坐标点进行dbscan聚类，得到dbscan_label,从0开始每个数字代表坐标点的类别，-1代表噪声
# 参照 https://www.cnblogs.com/pinard/p/6217852.html 的解释
print('DBSCAN...')
start_time = time.time()
dbscan_label_without_spy = DBSCAN(eps=0.25, min_samples=1).fit_predict(new_dim_points[0:num_each_cat * total_cat])
dbscan_label = DBSCAN(eps=3, min_samples=10).fit_predict(new_dim_points)  # eps 距离的阈值， min_sample，成为核心所需要的样本的阈值 2.5, 3

#assert len(np.unique(dbscan_label)) == total_cat
plt.figure()
plt.scatter(new_dim_points[0:(total_cat*num_each_cat), 0], new_dim_points[0:(total_cat*num_each_cat), 1], c=dbscan_label_without_spy)  # 画出dbscan分类后得到的所有坐标，一个颜色是一类
plt.show()
time_dif = get_time_dif(start_time)
print("Time usage:", time_dif)

##%%  math 2 dbscan
#for i in range(total_cat):
#    for j in range(len(dbscan_label)):
#        if i == dbscan_label[j]:
#            value_tbp = dbscan_label_without_spy[j]
#            np.place(dbscan_label_without_spy, dbscan_label_without_spy==value_tbp, [i + total_cat])
#            break
#for i in range(total_cat):
#    np.place(dbscan_label_without_spy, dbscan_label_without_spy==(i + total_cat), [i])
##%% spy_dbscan
#spy_each_cat = int(total_spy / total_cat)
#spy_8 = np.copy(dbscan_label[-spy_each_cat*2:-spy_each_cat])
#spy_16 = np.copy(dbscan_label[-spy_each_cat*5:-spy_each_cat*4])
#spy_32 = np.copy(dbscan_label[-spy_each_cat*4:-spy_each_cat*3])
#spy_64 = np.copy(dbscan_label[-spy_each_cat*3:-spy_each_cat*2])
#spy_Q = np.copy(dbscan_label[-spy_each_cat:])
#spy_Q_index = -1
#spy_8_index = -1
#spy_16_index = -1
#spy_32_index = -1
#spy_64_index = -1
#not_assigned = -1
#
##根据spy选最多cluster
##spy_8_index = np.bincount(spy_8[spy_8>=0]).argmax()
##spy_16_index = np.bincount(spy_16[spy_16>=0]).argmax()
##spy_32_index= np.bincount(spy_32[spy_32>=0]).argmax()
##spy_64_index = np.bincount(spy_64[spy_64>=0]).argmax()
##spy_Q_index = np.bincount(spy_Q[spy_Q>=0]).argmax()
#
##根据spy选最大cluster百分比
#spy_8_count = np.bincount(spy_8[spy_8>=0])
#spy_16_count = np.bincount(spy_16[spy_16>=0])
#spy_32_count = np.bincount(spy_32[spy_32>=0])
#spy_64_count = np.bincount(spy_64[spy_64>=0])
#spy_Q_count = np.bincount(spy_Q[spy_Q>=0])
#
#spy_8_percentage = np.zeros((5), np.int64)
#spy_16_percentage = np.zeros((5), np.int64)
#spy_32_percentage = np.zeros((5), np.int64)
#spy_64_percentage = np.zeros((5), np.int64)
#spy_Q_percentage = np.zeros((5), np.int64)
#
#spy_8_percentage[:len(spy_8_count)] = spy_8_count[:]
#spy_16_percentage[:len(spy_16_count)] = spy_16_count[:]
#spy_32_percentage[:len(spy_32_count)] = spy_32_count[:]
#spy_64_percentage[:len(spy_64_count)] = spy_64_count[:]
#spy_Q_percentage[:len(spy_Q_count)] = spy_Q_count[:]
#
#spy_8_sum = np.sum(spy_8_percentage)
#spy_16_sum = np.sum(spy_16_percentage)
#spy_32_sum = np.sum(spy_32_percentage)
#spy_64_sum = np.sum(spy_64_percentage)
#spy_Q_sum = np.sum(spy_Q_percentage)
#
#for i in range(total_cat):
#    temp = [spy_Q.tolist().count(i)/spy_Q_sum, spy_8.tolist().count(i)/spy_8_sum, spy_16.tolist().count(i)/spy_16_sum, spy_32.tolist().count(i)/spy_32_sum, spy_64.tolist().count(i)/spy_64_sum]
#    print(temp)
#    index = temp.index(max(temp))
#
#    if index == 0:
#        spy_Q_index = i
#    elif index == 1:
#        spy_8_index = i
#    elif index == 2:
#        spy_16_index = i
#    elif index == 3:
#        spy_32_index = i
#    elif index == 4:
#        spy_64_index = i
##
##temp = [spy_Q_index, spy_8_index, spy_16_index, spy_32_index, spy_64_index]        
##for i in range(total_cat):
##    if i not in temp:
##        not_assigned = i
##        
##if spy_Q_index == -1:
##    spy_Q_index = not_assigned
##elif spy_8_index == -1:
##    spy_8_index = not_assigned
##elif spy_16_index == -1:
##    spy_16_index = not_assigned
##elif spy_32_index == -1:
##    spy_32_index = not_assigned
##elif spy_64_index == -1:
##    spy_64_index = not_assigned
#
#
#
#
###根据cluster选最多spy
##for i in range(total_cat):
##    temp = [spy_Q.tolist().count(i), spy_8.tolist().count(i), spy_16.tolist().count(i), spy_32.tolist().count(i), spy_64.tolist().count(i)]
##    print(temp)
##    index = temp.index(max(temp))
##
##    if index == 0:
##        spy_Q_index = i
##    elif index == 1:
##        spy_8_index = i
##    elif index == 2:
##        spy_16_index = i
##    elif index == 3:
##        spy_32_index = i
##    elif index == 4:
##        spy_64_index = i
###
##temp = [spy_Q_index, spy_8_index, spy_16_index, spy_32_index, spy_64_index]        
##for i in range(total_cat):
##    if i not in temp:
##        not_assigned = i
##        
##if spy_Q_index == -1:
##    spy_Q_index = not_assigned
##elif spy_8_index == -1:
##    spy_8_index = not_assigned
##elif spy_16_index == -1:
##    spy_16_index = not_assigned
##elif spy_32_index == -1:
##    spy_32_index = not_assigned
##elif spy_64_index == -1:
##    spy_64_index = not_assigned
#
## %%使用dbscan_label把坐标点分类，目前是三类，如有更多类需要更改，（用于验证，实际使用可以不用）
#print('get points seperated by dbscan...')
#start_time = time.time()
#dbscan_8 = []
#dbscan_16 = []
#dbscan_32 = []
#dbscan_64 = []
#dbscan_Q = []
#counter = 0
#for j in range(total_cat):
#    #    assert j < total_cat
#    temp_set = []
#    counter = 0
#    for i in range(len(dbscan_label_without_spy)):
#        if dbscan_label_without_spy[i] == j:
#            temp_set.append(np.append(np.copy(new_dim_points[i]), [int(i)]))
#            counter += 1
#    print(counter)
#    #    assert counter == total_input/total_cat
#    if j == spy_16_index:
#        dbscan_16 = temp_set.copy()
#    if j == spy_32_index:
#        dbscan_32 = temp_set.copy()
#    if j == spy_64_index:
#        dbscan_64 = temp_set.copy()
#    if j == spy_8_index:
#        dbscan_8 = temp_set.copy()
#    if j == spy_Q_index:
#        dbscan_Q = temp_set.copy()
#    
#    dbscan_16 = np.array(dbscan_16)
#    dbscan_32 = np.array(dbscan_32)
#    dbscan_64 = np.array(dbscan_64)
#    dbscan_8 = np.array(dbscan_8)
#    dbscan_Q = np.array(dbscan_Q)
#time_dif = get_time_dif(start_time)
#print("Time usage:", time_dif)

#%%
## 根据centroid,centroid
#dbscan_8 = []
#dbscan_16 = []
#dbscan_32 = []
#dbscan_64 = []
#dbscan_Q = []
#for j in range(total_cat):
#    #    assert j < total_cat
#    temp_set = []
#    counter = 0
#    for i in range(len(dbscan_label_without_spy)):
#        if dbscan_label_without_spy[i] == j:
#            temp_set.append(np.append(np.copy(new_dim_points[i]), [int(i)]))
#            counter += 1
#    print(counter)
#    
#    if j == 0:
#        dbscan_0 = temp_set.copy()
#    if j == 1:
#        dbscan_1 = temp_set.copy()
#    if j == 2:
#        dbscan_2 = temp_set.copy()
#    if j == 3:
#        dbscan_3 = temp_set.copy()
#    if j == 4:
#        dbscan_4 = temp_set.copy()
#    
##dbscan_0 = np.array(dbscan_0)
##dbscan_1 = np.array(dbscan_1)
##dbscan_2 = np.array(dbscan_2)
##dbscan_3 = np.array(dbscan_3)
##dbscan_4 = np.array(dbscan_4)
#
#spy_each_cat = int(total_spy / total_cat)
#spy_16 = np.copy(new_dim_points[-spy_each_cat*5:-spy_each_cat*4])
#spy_32 = np.copy(new_dim_points[-spy_each_cat*4:-spy_each_cat*3])
#spy_64 = np.copy(new_dim_points[-spy_each_cat*3:-spy_each_cat*2])
#spy_8 = np.copy(new_dim_points[-spy_each_cat*2:-spy_each_cat])
#spy_Q = np.copy(new_dim_points[-spy_each_cat:])
#
#centroid_0 = get_centroid(dbscan_0)
#centroid_1 = get_centroid(dbscan_1)
#centroid_2 = get_centroid(dbscan_2)
#centroid_3 = get_centroid(dbscan_3)
#centroid_4 = get_centroid(dbscan_4)
#centroid_Q_spy = get_centroid(spy_Q)
#centroid_8_spy = get_centroid(spy_8)
#centroid_16_spy = get_centroid(spy_16)
#centroid_32_spy = get_centroid(spy_32)
#centroid_64_spy = get_centroid(spy_64)
#
#dbscan_array = [dbscan_0, dbscan_1, dbscan_2, dbscan_3, dbscan_4]
#dbscan_centroid = [centroid_0, centroid_1, centroid_2, centroid_3, centroid_4]
#spy_centorid = [centroid_Q_spy, centroid_8_spy, centroid_16_spy, centroid_32_spy, centroid_64_spy]
#
#for i in range(total_cat):
#    k = calucuate_closest_point(dbscan_centroid[i], spy_centorid)
#    print(k)
#    if k == 0:
#        dbscan_Q = dbscan_array[i].copy()
#    elif k == 1:
#        dbscan_8 = dbscan_array[i].copy()
#    elif k == 2:
#        dbscan_16 = dbscan_array[i].copy()
#    elif k == 3:
#        dbscan_32 = dbscan_array[i].copy()
#    elif k == 4:
#        dbscan_64 = dbscan_array[i].copy()
#        
#dbscan_Q = np.array(dbscan_Q)
#dbscan_8 = np.array(dbscan_8)
#dbscan_16 = np.array(dbscan_16)
#dbscan_32 = np.array(dbscan_32)
#dbscan_64 = np.array(dbscan_64)

# 根据centroid,distance
dbscan_8 = []
dbscan_16 = []
dbscan_32 = []
dbscan_64 = []
dbscan_Q = []
for j in range(total_cat):
    #    assert j < total_cat
    temp_set = []
    counter = 0
    for i in range(len(dbscan_label_without_spy)):
        if dbscan_label_without_spy[i] == j:
            temp_set.append(np.append(np.copy(new_dim_points[i]), [int(i)]))
            counter += 1
    print(counter)
    
    if j == 0:
        dbscan_0 = temp_set.copy()
    if j == 1:
        dbscan_1 = temp_set.copy()
    if j == 2:
        dbscan_2 = temp_set.copy()
    if j == 3:
        dbscan_3 = temp_set.copy()
    if j == 4:
        dbscan_4 = temp_set.copy()
    
#dbscan_0 = np.array(dbscan_0)
#dbscan_1 = np.array(dbscan_1)
#dbscan_2 = np.array(dbscan_2)
#dbscan_3 = np.array(dbscan_3)
#dbscan_4 = np.array(dbscan_4)

spy_each_cat = int(total_spy / total_cat)
spy_16 = np.copy(new_dim_points[-spy_each_cat*5:-spy_each_cat*4])
spy_32 = np.copy(new_dim_points[-spy_each_cat*4:-spy_each_cat*3])
spy_64 = np.copy(new_dim_points[-spy_each_cat*3:-spy_each_cat*2])
spy_8 = np.copy(new_dim_points[-spy_each_cat*2:-spy_each_cat])
spy_Q = np.copy(new_dim_points[-spy_each_cat:])

centroid_0 = get_centroid(dbscan_0)
centroid_1 = get_centroid(dbscan_1)
centroid_2 = get_centroid(dbscan_2)
centroid_3 = get_centroid(dbscan_3)
centroid_4 = get_centroid(dbscan_4)

centroid_Q = get_centroid(spy_Q)
centroid_8 = get_centroid(spy_8)
centroid_16 = get_centroid(spy_16)
centroid_32 = get_centroid(spy_32)
centroid_64 = get_centroid(spy_64)

dbscan_array = [dbscan_0, dbscan_1, dbscan_2, dbscan_3, dbscan_4]
dbscan_centroid = [centroid_0, centroid_1, centroid_2, centroid_3, centroid_4]
spy_array = [spy_Q, spy_8, spy_16, spy_32, spy_64]
spy_centroid = [centroid_Q, centroid_8, centroid_16, centroid_32, centroid_64]

#based on input
#for i in range(total_cat):
#    k = calculate_closest_points(dbscan_centroid[i], spy_array)
#    print(k)
#    if k == 0:
#        dbscan_Q = dbscan_array[i].copy()
#    elif k == 1:
#        dbscan_8 = dbscan_array[i].copy()
#    elif k == 2:
#        dbscan_16 = dbscan_array[i].copy()
#    elif k == 3:
#        dbscan_32 = dbscan_array[i].copy()
#    elif k == 4:
#        dbscan_64 = dbscan_array[i].copy()

#based on spy
matrix = []
distance = 0
for i in range(total_cat):
    distance = calculate_distance_centroid_v2(dbscan_centroid[i], dbscan_centroid)
    print(distance)
    k = calculate_percentage(dbscan_centroid[i], spy_array, distance)
    print(k)
    matrix.append(k)
matrix_max = np.array(matrix).argmax(axis=0)
for i in range(len(matrix_max)):
    if i == 0:
        index_Q = matrix_max[i]
        dbscan_Q = dbscan_array[index_Q].copy()
    elif i == 1:
        index_8 = matrix_max[i]
        dbscan_8 = dbscan_array[index_8].copy()
    elif i == 2:
        index_16 = matrix_max[i]
        dbscan_16 = dbscan_array[index_16].copy()
    elif i == 3:
        index_32 = matrix_max[i]
        dbscan_32 = dbscan_array[index_32].copy()
    elif i == 4:
        index_64 = matrix_max[i]
        dbscan_64 = dbscan_array[index_64].copy()

dbscan_Q = np.array(dbscan_Q)
dbscan_8 = np.array(dbscan_8)
dbscan_16 = np.array(dbscan_16)
dbscan_32 = np.array(dbscan_32)
dbscan_64 = np.array(dbscan_64)
# %% 通过label将tsne处理后得到的新的坐标的进行分类，目前是三类，（用于验证，实际使用可以不用）
print('get points seperated by label...')
start_time = time.time()
label_8 = []
label_16 = []
label_32 = []
label_64 = []
label_Q = []

for i in range(len(label_list)):
    if label_list[i][29:32] == '008':
        label_8.append(np.append(np.copy(new_dim_points[i]), [int(i)]))
    elif label_list[i][29:32] == '016':  # 根据文件名第29位进行识别，一下同理
        label_16.append(np.append(np.copy(new_dim_points[i]), [int(i)]))
    elif label_list[i][29:32] == '032':
        label_32.append(np.append(np.copy(new_dim_points[i]), [int(i)]))
    elif label_list[i][29:32] == '064':
        label_64.append(np.append(np.copy(new_dim_points[i]), [int(i)]))
    elif label_list[i][29:33] == 'QPSK':
        label_Q.append(np.append(np.copy(new_dim_points[i]), [int(i)]))

      
#label_16.extend(np.copy(new_dim_points[-80:-64]))
#label_32.extend(np.copy(new_dim_points[-64:-48]))
#label_64.extend(np.copy(new_dim_points[-48:-32]))
#label_8.extend(np.copy(new_dim_points[-32:-16]))
#label_Q.extend(np.copy(new_dim_points[-16:]))

label_8 = np.array(label_8)
label_16 = np.array(label_16)  # 变成numpy array
label_32 = np.array(label_32)
label_64 = np.array(label_64)
label_Q = np.array(label_Q)
time_dif = get_time_dif(start_time)
print("Time usage:", time_dif)

# %%
print('Plot TSNE by label...')
start_time = time.time()
for i in range(2):
    
    plt.figure()
    print_tsne(data=label_Q, label='Q', color='gold')
    print_tsne(data=label_8, label='8', color='green')
    print_tsne(data=label_16, label='16', color='dodgerblue')
    print_tsne(data=label_32, label='32', color='purple')
    print_tsne(data=label_64, label='64', color='red')    
    
    if i == 0:
        continue
    spy_8_position = np.copy(new_dim_points[-spy_each_cat*2:-spy_each_cat])
    spy_16_position = np.copy(new_dim_points[-spy_each_cat*5:-spy_each_cat*4])
    spy_32_position = np.copy(new_dim_points[-spy_each_cat*4:-spy_each_cat*3])
    spy_64_position = np.copy(new_dim_points[-spy_each_cat*3:-spy_each_cat*2])
    spy_Q_position = np.copy(new_dim_points[-spy_each_cat:])
    
    print_tsne(data=spy_8_position, label='8spy', color='green')
    print_tsne(data=spy_16_position, label='16spy', color='dodgerblue')
    print_tsne(data=spy_32_position, label='32spy', color='purple')
    print_tsne(data=spy_64_position, label='64spy', color='red')
    print_tsne(data=spy_Q_position, label='Qspy', color='gold')
    plt.show()
time_dif = get_time_dif(start_time)
print("Time usage:", time_dif)

# %% compute accuracy，将通过dbscan和label分类得到的类进行对比，得到算法正确率，incorrect 中包含错误识别的序号，可在label_list中寻找
incorrect = []
#    incorrect.extend(np.setdiff1d(dbscan_8[:, 2], label_8[:, 2]))
if dbscan_8.size == 0:
    correct_8 = 0
else:
    incorrect.extend(np.setdiff1d(label_8[:, 2], dbscan_8[:, 2]))
    incorrect = np.unique(np.array(incorrect))
    correct_8 = num_each_cat  - len(incorrect)

incorrect = []
#    incorrect.extend(np.setdiff1d(dbscan_16[:, 2], label_16[:, 2]))
if dbscan_16.size == 0:
    correct_16 = 0
else:        
    incorrect.extend(np.setdiff1d(label_16[:, 2], dbscan_16[:, 2]))
    incorrect = np.unique(np.array(incorrect))
    correct_16= num_each_cat  - len(incorrect)

incorrect = []
#    incorrect.extend(np.setdiff1d(dbscan_32[:, 2], label_32[:, 2]))
if dbscan_32.size == 0:
    correct_32 = 0
else:
    incorrect.extend(np.setdiff1d(label_32[:, 2], dbscan_32[:, 2]))
    incorrect = np.unique(np.array(incorrect))
    correct_32= num_each_cat  - len(incorrect)

incorrect = []
#    incorrect.extend(np.setdiff1d(dbscan_64[:, 2], label_64[:, 2]))
if dbscan_64.size == 0:
    correct_64 = 0
else:
    incorrect.extend(np.setdiff1d(label_64[:, 2], dbscan_64[:, 2]))
    incorrect = np.unique(np.array(incorrect))
    correct_64= num_each_cat  - len(incorrect)
                            
incorrect = []
#    incorrect.extend(np.setdiff1d(dbscan_Q[:, 2], label_Q[:, 2]))
if dbscan_Q.size == 0:
    correct_Q = 0
else:
    incorrect.extend(np.setdiff1d(label_Q[:, 2], dbscan_Q[:, 2]))
    incorrect = np.unique(np.array(incorrect))
    correct_Q= num_each_cat  - len(incorrect)


accuracy_8_t = correct_8 / num_each_cat
accuracy_16_t = correct_16 / num_each_cat
accuracy_32_t = correct_32 / num_each_cat
accuracy_64_t = correct_64 / num_each_cat
accuracy_Q_t = correct_Q / num_each_cat    
print('Accuracy_Q = ', accuracy_Q_t)
print('Accuracy_8 = ', accuracy_8_t)
print('Accuracy_16 = ', accuracy_16_t)
print('Accuracy_32 = ', accuracy_32_t)
print('Accuracy_64 = ', accuracy_64_t)

# %%
# print('Plot TSNE by dbscan...')
# start_time = time.time()
# plt.figure()
# print_tsne(data=dbscan_16, label='16', color='dodgerblue')
# print_tsne(data=dbscan_64, label='64', color='red')
# print_tsne(data=dbscan_Q, label='Q', color='gold')
# plt.show()
# time_dif = get_time_dif(start_time)
# print("Time usage:", time_dif)

# 总体时间
time_dif = get_time_dif(total_start_time)
print("Total time usage:", time_dif)

#data_util.remove_data(base_data_dir, num_each_cat)
