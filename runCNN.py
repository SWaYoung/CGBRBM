import tensorflow as tf
from cnn import CNN
import numpy as np
import os
import time
from datetime import timedelta
import pickle
import matplotlib.pyplot as plt


# 总体时间
#total_start_time = time.time()
#
## define functions
#def get_time_dif(start_time):
#    """获取已使用时间"""
#    end_time = time.time()
#    time_dif = end_time - start_time
#    return timedelta(seconds=int(round(time_dif)))


def create_raw_data(dirname, num_each_cat):  # 通过路径和文件名 创建x,y
    cats = []
    data = []
    label = []
    dirs = os.listdir(dirname)
    for i in dirs:
        temp = os.path.join(dirname, i)
        if os.path.isdir(temp) and i != 'test' and i != 'spy':
            cats.append(temp)
    iterator = iter(cats)
    counter = 0
    for i in range(len(cats)):
        dir_temp = next(iterator)
        for root, dirs, files in os.walk(dir_temp):
            print(root)
            for file in files[0:num_each_cat]:
                if os.path.splitext(file)[1] == '.jpg':
                    data.append(os.path.join(dir_temp, file))
                    label.append(counter)
        print("finished: ", dir_temp)
        counter += 1
    total_cat = int(len(cats))
    return np.array(data), np.array(label), total_cat


base_data_dir = '../Data/Image_Classification'  # 所有数据文件夹的根目录
chanel_increase = 16  # 第一层卷积后的chanel数量
filter_height = 5  # 卷积核尺寸
filter_width = 5  # 卷积核尺寸
offset_height = 80  # 截取图片的起始高度（从上至下）
offset_width = 270  # 截取图片的其实宽度（从左至右）
target_height = 700  # 截取图片的高度 700
target_width = 700  # 截取图片的宽度 700
resize_height = 64  # 缩放图片的高度
resize_width = 64  # 缩放图片的宽度
resize_method = 0  # 缩放图片的方式
num_each_cat = 1000  # 每类文件提取多少
n_epochs = 50
batch_size = 30  # 输入文件的batch数量
n_flatten = int(resize_height / 4 * resize_width / 4) * (chanel_increase * 2)  # 显层数量（自动计算）
n_hidden = 1000  # 隐层数量
shuffle = True
keep_prob = 1

x_train_raw, y_train_raw, total_cat = create_raw_data(base_data_dir, num_each_cat)

total_input = int(total_cat * num_each_cat)
cnn = CNN(offset_height,
          offset_width,
          target_height,
          target_width,
          resize_height,
          resize_width,
          resize_method,
          filter_height,
          filter_width,
          n_flatten,
          n_hidden,
          chanel_increase,
          total_cat,
          batch_size,
          n_epochs,
          total_input)

cnn.train(x_train_raw, y_train_raw, 1)


