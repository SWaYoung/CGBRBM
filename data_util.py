# coding: utf-8

import sys
from collections import Counter
import numpy as np
import tensorflow as tf
import os


def open_file(filename, mode='r'):
    return open(filename, mode, encoding='utf-8', errors='ignore')


def read_file(filename):
    contents, labels = [], []
    with open_file(filename) as f:
        for line in f:
            try:
                label, content = line.strip().split('\t')
                if content:
                    contents.append(list(content.split('/')))
                    labels.append(label)
            except:
                pass
    return contents, labels  # contents: 两维list，包含所有句子，每个句子一个list；labels：list，类


def create_raw_data(dirname, ):
    lables = []
    datas = []
    quam16_dir = os.path.join(dirname, '16QAM')
    quam64_dir = os.path.join(dirname, '64QAM')
    qpsk_dir = os.path.join(dirname, 'QPSK')
    types = iter([quam16_dir, quam64_dir, qpsk_dir])
    for i in range(3):
        dir_temp = next(types)
        for root, dirs, files in os.walk(dir_temp):
            print(root)
            for file in files[0:1000]:
                if os.path.splitext(file)[1] == '.jpg':
                    lables.append(os.path.join(dir_temp, file))
                    datas.append(os.path.join(dir_temp, file))
        print("finished: ", dir_temp)
    return np.array(datas), np.array(lables)

    # def build_image_dataset(input_x,
#                         input_y,
#                         offset_height=50,
#                         offset_width=250,
#                         target_height=750,
#                         target_width=750,
#                         resize_height=256,
#                         resize_width=256,
#                         resize_method=3):  # 0:BILINEAR; 1:NEAREST_NEIGHBOR; 2:BICUBIC; 3:AREA
#     image = tf.read_file(input_x)
#     decoded_image = tf.image.decode_jpeg(image)
#     cropped_image = tf.image.crop_to_bounding_box(decoded_image, offset_height, offset_width, target_height, target_width)
#     resized_image = tf.image.resize_images(cropped_image,[resize_height, resize_width], resize_method)
#     gray_image = tf.image.rgb_to_grayscale(resized_image)
#
#     return gray_image, input_y
#
#
# def process_image(data_x,  # 图片路径，np array
#                   data_y,  # label, np array
#                   n_epoches=10,
#                   batch_size=10,
#                   offset_height=50,
#                   offset_width=250,
#                   target_height=750,
#                   target_width=750,
#                   resize_height=256,
#                   resize_width=256,
#                   resize_method=3,
#                   shuffle=True, ):
#     n_data = data_x.shape[0]
#
#     if shuffle:
#         dataset = tf.data.Dataset.from_tensor_slices((data_x, data_y)).map(
#             lambda x, y: build_image_dataset(x, y, offset_height=offset_height,
#                                              offset_width=offset_width,
#                                              target_height=target_height,
#                                              target_width=target_width,
#                                              resize_height=resize_height,
#                                              resize_width=resize_width,
#                                              resize_method=resize_method)).shuffle(n_data,
#                                                                                    reshuffle_each_iteration=True).batch(
#             batch_size).repeat(n_epoches)
#     else:
#         dataset = tf.data.Dataset.from_tensor_slices((data_x, data_y)).map(
#             lambda x, y: build_image_dataset(x, y, offset_height=offset_height,
#                                              offset_width=offset_width,
#                                              target_height=target_height,
#                                              target_width=target_width,
#                                              resize_height=resize_height,
#                                              resize_width=resize_width,
#                                              resize_method=resize_method)).batch(batch_size).repeat(n_epoches)
#
#     iterator = dataset.make_initializable_iterator()
#     return iterator
