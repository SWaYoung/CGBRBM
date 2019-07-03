import tensorflow as tf
import os
import matplotlib.pyplot as plt
import numpy as np
import pickle


# 用于测试裁剪、缩放图片
# def file_name(file_dir):  # 特定类型的文件
#     L = []
#     for root, dirs, files in os.walk(file_dir):
#         print(root)
#         for file in files:
#             if os.path.splitext(file)[1] == '.jpg':
#                 L.append(os.path.join(root, file))
#     return L


def read(filename, label):
    crop_box = [80, 270, 700, 700]
    image_string = tf.read_file(filename)
    decode_image = tf.image.decode_jpeg(image_string)
    crop_image = tf.image.crop_to_bounding_box(decode_image, crop_box[0], crop_box[1], crop_box[2], crop_box[3])
    resize_image = tf.image.resize_images(crop_image, [64, 64], 3)
    gray_image = tf.image.rgb_to_grayscale(resize_image)
    normalized_image = tf.image.per_image_standardization(gray_image)
    normalized_image = tf.squeeze(normalized_image)
    return normalized_image, label


path = np.array(['test/1.jpg', 'test/2.jpg'])
label = np.array(['test/1.jpg', 'test/2.jpg'])
path = np.array(path)
label = np.array(label)

# path = file_name('test')
# label = file_name('test')

xp = tf.placeholder(tf.string, [None], name='input')
file_queue = tf.data.Dataset.from_tensor_slices((xp, label))
file_queue = file_queue.map(read).shuffle(2)

iterator = file_queue.make_initializable_iterator()
result_it, label_it = iterator.get_next()

with tf.Session() as sess:
    sess.run(iterator.initializer, feed_dict={xp: path})
    value1 = sess.run([result_it, label_it])
    value2 = sess.run([result_it, label_it])

plt.figure
plt.imshow(value1[0], cmap='gray')
plt.show()

if tf.test.gpu_device_name():
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
else:
    print("Please install GPU version of TF")
    

writer = tf.summary.FileWriter('.')
writer.add_graph(tf.get_default_graph())

# 改名
import os
folders = os.listdir('.')
for folder in folders:
    file_root = os.path.join('.', folder)
    files = os.listdir(file_root)
    counter = 1
    for file in files:
        new_name = folder + '_' + str(counter) + '.jpg'
        os.rename(os.path.join(file_root, file), os.path.join('.', new_name))
        counter +=1
#
import shutil
import os
OSNR = 15
for i in range(100):
    shutil.copy2('064QAM/o.jpg', '064QAM/64QAM{}_{}.jpg'.format(OSNR, i+1))
os.remove('064QAM/o.jpg')
for i in range(100):
    shutil.copy2('032QAM/o.jpg', '032QAM/32QAM{}_{}.jpg'.format(OSNR, i+1))
os.remove('032QAM/o.jpg')
for i in range(100):
    shutil.copy2('016QAM/o.jpg', '016QAM/16QAM{}_{}.jpg'.format(OSNR, i+1))
os.remove('016QAM/o.jpg')
for i in range(100):
    shutil.copy2('008QAM/o.jpg', '008QAM/8QAM{}_{}.jpg'.format(OSNR, i+1))
os.remove('008QAM/o.jpg')
for i in range(100):
    shutil.copy2('QPSK/o.jpg', 'QPSK/QPSK{}_{}.jpg'.format(OSNR, i+1))
os.remove('QPSK/o.jpg')


        
