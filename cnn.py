import tensorflow as tf


class CNN(object):
    def __init__(self,
                 offset_height,
                 offset_width,
                 target_height,
                 target_width,
                 resize_height,
                 resize_width,
                 resize_method,  # 0:BILINEAR; 1:NEAREST_NEIGHBOR; 2:BICUBIC; 3:AREA
                 filter_height,  # 卷积核尺寸
                 filter_width,  # 卷积核尺寸
                 n_flatten,
                 n_hidden,
                 chanel_increase,
                 total_cat,
                 batch_size,
                 n_epochs,
                 total_input,
                 shuffle=True):
        self.filter_sizes = [filter_height, filter_width]
        self.offset_height = offset_height
        self.offset_width = offset_width
        self.target_height = target_height
        self.target_width = target_width
        self.resize_height = resize_height
        self.resize_width = resize_width
        self.resize_method = resize_method
        self.n_epochs = n_epochs
        filter_sizes = [filter_height, filter_width]

        self.raw_input_x = tf.placeholder(tf.string, [total_input], name="raw_input_x")
        self.raw_input_y = tf.placeholder(tf.int32, [total_input], name="raw_input_y")
        # self.input_x = tf.placeholder(tf.float32, [None, None, None, None], name='input_x')
        # self.input_y = tf.placeholder(tf.float32, [None, num_classes], name='input_y')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.image_batch_get_next = None
        self.label_batch_get_next = None
        self.iterator = None
        self.label = None

        with tf.name_scope('process-images'):
            if shuffle:
                dataset = tf.data.Dataset.from_tensor_slices((self.raw_input_x, self.raw_input_y)).shuffle(total_input,
                                                                                                 reshuffle_each_iteration=True).map(
                    lambda x, y: self.build_image_dataset(x, y), num_parallel_calls=4).batch(batch_size).repeat(
                    self.n_epochs).prefetch(1)
            else:
                dataset = tf.data.Dataset.from_tensor_slices((self.raw_input_x, self.raw_input_y)).map(
                    lambda x, y: self.build_image_dataset(x, y), num_parallel_calls=4).batch(batch_size).repeat(
                    self.n_epochs).prefetch(1)

            self.iterator = dataset.make_initializable_iterator()
            self.image_batch_get_next, self.label_batch_get_next = self.iterator.get_next()
            self.label = tf.one_hot(self.label_batch_get_next, total_cat, axis=-1)

        with tf.name_scope('conv-maxpool'):
            filter_shape = filter_sizes.copy()
            filter_shape.extend([1, chanel_increase])
            # 第一层conv的filter
            self.c_w1 = tf.get_variable(name="c_w1", shape=filter_shape,
                                        initializer=tf.initializers.truncated_normal(stddev=0.1))
            # 第一层conv的bias
            self.c_b1 = tf.get_variable(name='c_b1', initializer=tf.constant(0.1, shape=[chanel_increase]))
            filter_shape = filter_sizes.copy()
            filter_shape.extend([chanel_increase, chanel_increase * 2])
            # 第二层conv的filter
            self.c_w2 = tf.get_variable(name="c_w2", shape=filter_shape,
                                        initializer=tf.initializers.truncated_normal(stddev=0.1))
            # 第二层conv的bias
            self.c_b2 = tf.get_variable(name='c_b2', initializer=tf.constant(0.1, shape=[chanel_increase * 2]))

            # 第一层conv+pool
            self.conv1 = tf.nn.conv2d(self.image_batch_get_next, self.c_w1, strides=[1, 1, 1, 1], padding="SAME", name="conv1")
            self.relu1 = tf.nn.relu(tf.nn.bias_add(self.conv1, self.c_b1), name='relu1')
            self.pool1 = tf.nn.max_pool(self.relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME",
                                        name="pool1")

            # 第二层conv+pool
            self.conv2 = tf.nn.conv2d(self.pool1, self.c_w2, strides=[1, 1, 1, 1], padding="SAME", name="conv2")
            self.relu2 = tf.nn.relu(tf.nn.bias_add(self.conv2, self.c_b2), name='relu2')
            self.pool2 = tf.nn.max_pool(self.relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME",
                                        name="pool2")

        with tf.name_scope('fully-connected-input'):
            # fc输入层的weight
            self.f_w1 = tf.get_variable(name='f_w1', shape=[n_flatten, n_hidden],
                                        initializer=tf.initializers.truncated_normal(stddev=0.1))
            # fc输入层的bias
            self.f_b1 = tf.get_variable(name='f_b1', initializer=tf.constant(0.1, shape=[n_hidden]))

            # fully connected layer: input
            self.flatten = tf.reshape(self.pool2, [-1, n_flatten])
            # fc_input = tf.nn.relu(tf.nn.bias_add(tf.matmul(self.flatten, self.f_w1), self.f_b1), name='fc_input')
            self.fc_input = tf.nn.relu(tf.nn.xw_plus_b(self.flatten, self.f_w1, self.f_b1), name='fc_input')

            # Add dropout
        with tf.name_scope("dropout"):
            self.drop = tf.nn.dropout(self.fc_input, self.dropout_keep_prob, name='dropout')

        with tf.name_scope('fully-connected-output'):
            # fc输出层的weight
            self.f_w2 = tf.get_variable(name='f_w2', shape=[n_hidden, total_cat],
                                        initializer=tf.initializers.truncated_normal(stddev=0.1))
            # fc输出层的bias
            self.f_b2 = tf.get_variable(name='f_b2', initializer=tf.constant(0.1, shape=[total_cat]))

            # fully connected laye: output
            # fc_output = tf.nn.softmax(tf.nn.bias_add(tf.matmul(self.drop, self.f_w2), self.f_b2), name='fc_output')
            self.fc_output = tf.nn.xw_plus_b(self.drop, self.f_w2, self.f_b2, name='output')

        with tf.name_scope('accuracy'):
            correct_predictions = tf.equal(tf.argmax(self.fc_output, 1), tf.argmax(self.label, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32), name='accuracy')

        with tf.name_scope('calculate-loss'):
            self.loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.label, logits=self.fc_output), name='loss')

        with tf.name_scope('optimizer'):
            self.train_step = tf.train.AdamOptimizer(1e-4).minimize(self.loss)

        init = tf.global_variables_initializer()
        session_conf = tf.ConfigProto(allow_soft_placement=False, log_device_placement=True)
        session_conf.gpu_options.allow_growth = True
        self.sess = tf.Session(config=session_conf)
        self.sess.run(init)

    def build_image_dataset(self, input_x,
                            input_y):
        image = tf.read_file(input_x)
        decoded_image = tf.image.decode_jpeg(image)
        cropped_image = tf.image.crop_to_bounding_box(decoded_image, self.offset_height, self.offset_width,
                                                      self.target_height, self.target_width)
        resized_image = tf.image.resize_images(cropped_image, [self.resize_height, self.resize_width],
                                               self.resize_method)
        gray_image = tf.image.rgb_to_grayscale(resized_image)
        normalized_image = tf.image.per_image_standardization(gray_image)
        return normalized_image, input_y

    def train(self,
              raw_data_x,
              raw_data_y,
              keep_prob):
        self.sess.run(self.iterator.initializer, feed_dict={self.raw_input_x: raw_data_x, self.raw_input_y: raw_data_y})
        for i in range(self.n_epochs):
            _, loss = self.sess.run([self.train_step, self.loss], feed_dict={self.dropout_keep_prob: keep_prob})
            print('Epoch: {0:02d}, loss: {1:.4f}'.format(i, loss))

    def predict(self,
                raw_data_x,
                raw_data_y,
                keep_prob,
                batch_size):
        self.sess.run(self.iterator.initializer, feed_dict={self.raw_input_x: raw_data_x, self.raw_input_y: raw_data_y})
        # To Be Continued...
