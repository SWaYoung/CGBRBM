# coding: utf-8

# import
import tensorflow as tf
import numpy as np
import sys
from .gbrbm import GBRBM
import pickle


#
class CGBRBM(GBRBM):
    def __init__(self,
                 n_visible=8192,
                 n_hidden=800,
                 chanel_increase=16,
                 filter_sizes=[5, 5],  # 卷积核尺寸
                 **kwargs):

        # n_visible                 显层个数
        # n_hidden                  隐层个数

        # chanel_increase           chanel 第一层之后的增加个数，第二层之后每次变两倍
        # filter_sizes = [5， 5]     卷积核尺寸

        self.n_visible = n_visible
        self.matrix_counter = 0
        self.chanel_increase = chanel_increase
        self.filter_sizes = filter_sizes
        self.raw_input_x = tf.placeholder(tf.string, [None], name="input_x")
        self.raw_input_y = tf.placeholder(tf.string, [None], name="input_y")
        self.conv_input = tf.placeholder(dtype=tf.float32, shape=[None, None, None, None], name="input")

        filter_shape = self.filter_sizes.copy()
        filter_shape.extend([1, self.chanel_increase])
        self.w1 = tf.get_variable(name="w1", shape=filter_shape,
                                  initializer=tf.initializers.truncated_normal(stddev=0.1))
        self.b1 = tf.get_variable(name='b1', initializer=tf.constant(0.1, shape=[self.chanel_increase]))
        filter_shape = self.filter_sizes.copy()
        filter_shape.extend([self.chanel_increase, self.chanel_increase * 2])
        self.w2 = tf.get_variable(name="w2", shape=filter_shape,
                                  initializer=tf.initializers.truncated_normal(stddev=0.1))
        self.b2 = tf.get_variable(name='b2', initializer=tf.constant(0.1, shape=[self.chanel_increase * 2]))

        GBRBM.__init__(self, n_visible, n_hidden, **kwargs)

    def build_convolution_max_pooling(self):
        # convolution and max-pooling layers

        pooled_outputs = []
        filter_shape = self.filter_sizes.copy()
        filter_shape.extend([1, self.chanel_increase])
        conv1 = tf.nn.conv2d(self.conv_input, self.w1, strides=[1, 1, 1, 1], padding="SAME", name="conv1")
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, self.b1), name='relu1')
        pool1 = tf.nn.max_pool(relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME", name="pool1")

        filter_shape = self.filter_sizes.copy()
        filter_shape.extend([self.chanel_increase, self.chanel_increase * 2])
        conv2 = tf.nn.conv2d(pool1, self.w2, strides=[1, 1, 1, 1], padding="SAME", name="conv2")
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, self.b2), name='relu2')
        pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME", name="pool2")

        self.flatten = tf.reshape(pool2, [-1, self.n_visible])

    @staticmethod
    def build_image_dataset(input_x,
                            input_y,
                            offset_height=80,
                            offset_width=270,
                            target_height=700,
                            target_width=700,
                            resize_height=64,
                            resize_width=64,
                            resize_method=0):  # 0:BILINEAR; 1:NEAREST_NEIGHBOR; 2:BICUBIC; 3:AREA
        image = tf.read_file(input_x)
        decoded_image = tf.image.decode_jpeg(image)
        cropped_image = tf.image.crop_to_bounding_box(decoded_image, offset_height, offset_width, target_height,
                                                      target_width)
        resized_image = tf.image.resize_images(cropped_image, [resize_height, resize_width], resize_method)
        gray_image = tf.image.rgb_to_grayscale(resized_image)
        one_hot = tf.image.per_image_standardization(gray_image)
        return one_hot, input_y

    def cfit(self,
             raw_data_x,  # 图片路径，np array
             raw_data_y,  # label, np array
             n_epoches=10,
             batch_size=30,
             offset_height=80,
             offset_width=270,
             target_height=700,
             target_width=700,
             resize_height=64,
             resize_width=64,
             resize_method=0,
             shuffle=True,
             verbose=True):
        assert n_epoches > 0

        n_data = raw_data_x.shape[0]

        if batch_size > 0:
            n_batches = n_data // batch_size + (0 if n_data % batch_size == 0 else 1)
        else:
            n_batches = 1

        if shuffle:
            dataset = tf.data.Dataset.from_tensor_slices((self.raw_input_x, self.raw_input_y)).map(
                lambda x, y: self.build_image_dataset(x, y, offset_height=offset_height,
                                                      offset_width=offset_width,
                                                      target_height=target_height,
                                                      target_width=target_width,
                                                      resize_height=resize_height,
                                                      resize_width=resize_width,
                                                      resize_method=resize_method)).shuffle(n_data,
                                                                                            reshuffle_each_iteration=True).batch(
                batch_size).repeat(n_epoches)
        else:
            dataset = tf.data.Dataset.from_tensor_slices((self.raw_input_x, self.raw_input_y)).map(
                lambda x, y: self.build_image_dataset(x, y, offset_height=offset_height,
                                                      offset_width=offset_width,
                                                      target_height=target_height,
                                                      target_width=target_width,
                                                      resize_height=resize_height,
                                                      resize_width=resize_width,
                                                      resize_method=resize_method)).batch(batch_size).repeat(n_epoches)

        iterator = dataset.make_initializable_iterator()
        image_batch_get_next, label_batch_get_next = iterator.get_next()
        self.sess.run(iterator.initializer, feed_dict={self.raw_input_x: raw_data_x, self.raw_input_y: raw_data_y})

        errs = []

        file_counter = 0
        for e in range(n_epoches):
            if verbose and not self._use_tqdm:
                print('Epoch: {:d}'.format(e))

            epoch_errs = np.zeros((n_batches,))
            epoch_errs_ptr = 0

            r_batches = range(n_batches)

            if verbose and self._use_tqdm:
                r_batches = self._tqdm(r_batches, desc='Epoch: {:d}'.format(e), ascii=True, file=sys.stdout)

            for _ in r_batches:
                feature_temp, label_temp = self.sess.run([image_batch_get_next, label_batch_get_next])
                convolved_images = self.sess.run(self.flatten, feed_dict={self.conv_input: feature_temp})
                if file_counter == 0:
                    with open('./pickle/feature_obj' + str(self.matrix_counter) + '.pkl', 'wb') as p:
                        pickle.dump(convolved_images, p)
                    with open('./pickle/label_obj' + str(self.matrix_counter) + '.pkl', 'wb') as p:
                        pickle.dump(label_temp, p)
                    self.matrix_counter += 1

                self.partial_fit(convolved_images)
                batch_err = self.get_err(convolved_images)
                epoch_errs[epoch_errs_ptr] = batch_err
                epoch_errs_ptr += 1

            if verbose:
                err_mean = epoch_errs.mean()
                if self._use_tqdm:
                    self._tqdm.write('Train error: {:.4f}'.format(err_mean))
                    self._tqdm.write('')
                else:
                    print('Train error: {:.4f}'.format(err_mean))
                    print('')
                sys.stdout.flush()

            errs = np.hstack([errs, epoch_errs])
            file_counter += 1
        return errs

    def ctransform(self,
                   data_x,
                   batch_size=30):
        n_data = data_x.shape[0]

        if batch_size > 0:
            n_batches = n_data // batch_size + (0 if n_data % batch_size == 0 else 1)
        else:
            n_batches = 1

        outputs = np.zeros((n_data, self.n_hidden))
        output_counter = 0
        data_x_cpy = data_x

        r_batches = range(n_batches)
        for b in r_batches:
            batch_x_temp = data_x_cpy[b * batch_size:(b + 1) * batch_size]
            output = self.transform(batch_x_temp)
            for i in range(len(output)):
                outputs[output_counter] = output[i]
                output_counter += 1

        return outputs

    def cget_error(self,
                   data_x,
                   batch_size=30,
                   offset_height=50,
                   offset_width=250,
                   target_height=750,
                   target_width=750,
                   resize_height=256,
                   resize_width=256,
                   resize_method=3):
        n_data = data_x.shape[0]

        if batch_size > 0:
            n_batches = n_data // batch_size + (0 if n_data % batch_size == 0 else 1)
        else:
            n_batches = 1

        errors = np.zeros((n_batches,))
        error_counter = 0

        dataset = tf.data.Dataset.from_tensor_slices(self.raw_input_x).map(
            lambda x, y: self.build_image_dataset(x, y, offset_height=offset_height,
                                                  offset_width=offset_width,
                                                  target_height=target_height,
                                                  target_width=target_width,
                                                  resize_height=resize_height,
                                                  resize_width=resize_width,
                                                  resize_method=resize_method)).batch(batch_size)

        iterator = dataset.make_initializable_iterator()
        batch_x = iterator.get_next()
        self.sess.run(iterator.initializer, feed_dict={self.raw_input_x: data_x})

        r_batches = range(n_batches)
        for _ in r_batches:
            batch_x = self.sess.run(self.flatten, feed_dict={self.conv_input: self.sess.run(batch_x)})
            error = self.get_err(batch_x)
            errors[error_counter] = error
            error_counter += 1

        return errors
