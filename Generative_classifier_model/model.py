from data import input_setup
from plot import make_plot

import time
import os
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from scipy.spatial import distance
from scipy import linalg

import numpy as np
import tensorflow as tf

# Set random seed for data permutation
np.random.seed(0)
# Set random seed for tf.Variable initialization
tf.set_random_seed(1234)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class SRCNN(object):

    def __init__(self,
                 sess,
                 image_size=32,
                 label_size=32,
                 batch_size=128,
                 c_dim=1,
                 checkpoint_dir=None,
                 sample_dir=None):

        self.sess = sess
        self.is_grayscale = (c_dim == 1)
        self.image_size = image_size
        self.label_size = label_size
        self.batch_size = batch_size

        self.c_dim = c_dim

        self.mu = 0
        self.sigma = 0.1
        self.layer_depth = {'L1': 6, 'L2': 16, 'L3': 128, 'L4': 64, 'L5': 10}

        self.checkpoint_dir = checkpoint_dir
        self.sample_dir = sample_dir
        self.build_model()


    def build_model(self):
        self.images = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, self.c_dim], name='images')
        self.labels = tf.placeholder(tf.int32, shape=[None], name='labels')
        self.one_hot = tf.one_hot(self.labels, 10)
        self.keep_prob = tf.placeholder(tf.float32)

        layer_depth = self.layer_depth
        self.weights = {
            'conv1_w': tf.Variable(
            np.sqrt(2.0 / 32 * 32) * tf.truncated_normal(shape=[5, 5, 1, layer_depth.get('L1')],
                                                        mean=self.mu, stddev=self.sigma),name='conv1_w'),
            'conv2_w': tf.Variable(
            np.sqrt(2.0 / 14 * 14) * tf.truncated_normal(shape=[5, 5, layer_depth.get('L1'), layer_depth.get('L2')],
                                                         mean=self.mu, stddev=self.sigma), name='conv2_w'),
            'fullc1_w': tf.Variable(
            np.sqrt(2.0 / 400) * tf.truncated_normal(shape=(400, layer_depth.get('L3')),
                                                    mean=self.mu, stddev=self.sigma),name='fullc1_w'),
            'fullc2_w': tf.Variable(
            np.sqrt(2.0 / 120) * tf.truncated_normal(shape=(layer_depth.get('L3'), layer_depth.get('L4')), mean=self.mu,
                                                     stddev=self.sigma), name='fullc2_w'),
            'fullc3_w': tf.Variable(
            tf.truncated_normal(shape=(layer_depth.get('L4'), layer_depth.get('L5')), mean=self.mu, stddev=self.sigma),
            name='fullc3_w')

        }
        self.biases = {
            'conv1_b': tf.Variable(tf.zeros(layer_depth.get('L1')), name='conv1_b'),
            'conv2_b': tf.Variable(tf.zeros(layer_depth.get('L2')), name='conv2_b'),
            'fullc1_b': tf.Variable(tf.zeros(layer_depth.get('L3')), name='fullc1_b'),
            'fullc2_b': tf.Variable(tf.zeros(layer_depth.get('L4')), name='fullc2_b'),
            'fullc3_b': tf.Variable(tf.zeros(layer_depth.get('L5')), name='fullc3_b')
        }

        self.pred = self.model()

        # Loss function (MSE)
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.pred, labels=self.one_hot))

        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.pred, 1), tf.argmax(self.one_hot, 1)), tf.float32))

        self.saver = tf.train.Saver()

    def train(self, config):

        train_data, valid_data, train_label, valid_label = input_setup(config)

        # Stochastic gradient descent with the standard backpropagation
        self.train_op = tf.train.AdamOptimizer(learning_rate=config.learning_rate).minimize(self.loss)

        tf.global_variables_initializer().run()

        counter = 0
        start_time = time.time()

        if self.load(self.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        if config.is_train:
            print("Training...")

            # Tensor Board
            #tf.summary.scalar("loss", self.loss)
            #tf.summary.scalar("accuracy", self.accuracy)
            #summary = tf.summary.merge_all()

            #writer = tf.summary.FileWriter('./logs')
            #writer.add_graph(self.sess.graph)

            for ep in range(config.epoch):
                # Run by batch images
                train_data, train_label = shuffle(train_data, train_label)
                batch_idxs = len(train_data) // config.batch_size
                for idx in range(0, batch_idxs):
                    batch_images = train_data[idx * config.batch_size: (idx + 1) * config.batch_size]
                    batch_labels = train_label[idx * config.batch_size: (idx + 1) * config.batch_size]
                    counter += 1
                    self.sess.run(self.train_op,
                                               feed_dict={self.images: batch_images, self.labels: batch_labels, self.keep_prob: 0.8})
                    #writer.add_summary(s, global_step=ep)

                    #if ep % 10 == 0 and counter % 40 == 0:
                loss = self.loss.eval(session=self.sess, feed_dict={self.images: train_data, self.labels: train_label,
                                                                    self.keep_prob: 1.0})
                acc = self.accuracy.eval(session=self.sess,
                                         feed_dict={self.images: train_data, self.labels: train_label,
                                                    self.keep_prob: 1.0})
                print("Epoch: [%2d], step: [%2d], time: [%4.4f], loss: [%.8f], accuracy: [%.2f%%]" \
                              % ((ep + 1), counter, time.time() - start_time, loss, acc*100))

            self.save(config.checkpoint_dir, counter)
            valid_acc = self.accuracy.eval(session=self.sess, feed_dict={self.images: valid_data, self.labels: valid_label, self.keep_prob: 1.0})
            print(f'Validation set Accuracy: {valid_acc*100:.2f}%')

        else:
            print("Testing...")
            valid_acc = self.accuracy.eval(session=self.sess,
                                           feed_dict={self.images: valid_data, self.labels: valid_label, self.keep_prob: 1.0})
            test_acc = self.accuracy.eval(session=self.sess,
                                          feed_dict={self.images: train_data, self.labels: train_label, self.keep_prob: 1.0})
            print(f'Validation set Accuracy: {valid_acc * 100:.2f}%')
            print(f'Test set Accuracy: {test_acc * 100:.2f}%')
            self.Cal_Distance(train_data, train_label,config)

    def model(self):
        global conv1, conv2, fullc1, fullc2
        # L1: Convolutional ~ (32, 32, 1) -> (28, 28, 6)
        conv1 = tf.nn.conv2d(self.images, self.weights['conv1_w'], strides=[1, 1, 1, 1], padding='VALID') + self.biases['conv1_b']
        conv1 = tf.nn.relu(conv1)
        conv1 = tf.nn.dropout(conv1, self.keep_prob)
        # Pooling ~ (28, 28, 6) -> (14, 14, 6)
        pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

        # L2: Convolutional ~ (14, 14, 6) -> (10, 10, 16)
        conv2 = tf.nn.conv2d(pool1, self.weights['conv2_w'], strides=[1, 1, 1, 1], padding='VALID') + self.biases['conv2_b']
        conv2 = tf.nn.relu(conv2)
        conv2 = tf.nn.dropout(conv2, self.keep_prob)
        # Pooling ~ (10, 10, 16) -> (5, 5, 16)
        pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

        # Flatten ~ (5, 5, 16) -> (400)
        fullc1 = tf.contrib.layers.flatten(pool2)

        # L3: Fully-connected ~ (400) -> (128)
        fullc1 = tf.matmul(fullc1, self.weights['fullc1_w']) + self.biases['fullc1_b']
        fullc1 = tf.nn.relu(fullc1)
        fullc1 = tf.nn.dropout(fullc1, self.keep_prob)

        # L4: Fully-connected ~ (128) -> (64)
        fullc2 = tf.matmul(fullc1, self.weights['fullc2_w']) + self.biases['fullc2_b']
        fullc2_ = tf.nn.relu(fullc2)
        fullc2_ = tf.nn.dropout(fullc2_, self.keep_prob)

        # L5: Fully-connected ~ (64) -> (10)
        # logits(outputs): 입력데이터를 네트워크에서 feed-forward를 통해 나온 결과물
        logits = tf.matmul(fullc2_, self.weights['fullc3_w']) + self.biases['fullc3_b']

        return logits

    def save(self, checkpoint_dir, step):
        model_name = "Mnist.model"
        model_dir = "%s_%s" % ("mnist", self.label_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoints...")
        model_dir = "%s_%s" % ("mnist", self.label_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False

    def get_Mu_hat(self, data, num_neurons, num_labels):
        mu_hat = [np.zeros(num_neurons)] * num_labels
        for label in range(num_labels):
            mu_hat[label] = np.mean(data[label], axis=0)
        return mu_hat

    def get_Sigma_hat(self, data, mu_hat, num_neurons, num_labels, num_data):
        sigma_hat = np.zeros([num_neurons, num_neurons])
        for label in range(num_labels):
            for datum in data[label]:
                u = np.reshape(datum, (num_neurons, 1))
                v = np.reshape(mu_hat[label], (num_neurons, 1))
                sigma_hat += np.matmul((u - v), (u - v).T)
        sigma_hat /= num_data
        return sigma_hat

    def Cal_mahalanobis_max(self, m_dist_data, num_labels):
        temp = [None] * num_labels
        for label in range(num_labels):
            temp[label] = list()
        for label in range(num_labels):
            temp[label].append(np.percentile(m_dist_data[label], 100, interpolation='linear'))
        m_max = np.array(temp)
        return m_max

    def Cal_Distance(self, test_data, test_label, config):
        # Generative classifier =======================================================================================
        # Data(N = 30000, data[i][j] = (j+1)th data of label i): get f(x[i]) and append distinguished by label --------
        config.is_train = True  # for getting training data
        train_data, valid_data, train_label, valid_label = input_setup(config)

        label_x = np.array(self.sess.run(tf.argmax(self.pred, 1),
                                    feed_dict={self.images: train_data, self.keep_prob: 1.0}))  # label of data x

        f_x = np.array(self.sess.run(fullc2, feed_dict={self.images: train_data, self.keep_prob: 1.0
                                                        }))  # output of the hidden layer of data x

        label_test = np.array(self.sess.run(tf.argmax(self.pred, 1),
                                         feed_dict={self.images: train_data, self.keep_prob: 1.0}))  # label of data x
        test_f_x = self.sess.run(fullc2, feed_dict={self.images: test_data, self.keep_prob: 1.0}) # output of the hidden layer of test_data

        N = len(train_data)
        num_labels = 10  # number of labels: 10
        num_neurons = self.layer_depth.get('L4')  # number of neurons(dimensions) in the hidden layer(64)
        temp_penult = [None] * num_labels
        for label in range(num_labels):
            temp_penult[label] = list()
        for data_num in range(len(train_data)):
            temp_penult[label_x[data_num]].append(f_x[data_num])
        penultimate_data = np.array(temp_penult)  # penultimate_data[0] ~ [9]까지 각각 레이블에 상응하는 penultimate layer의 값들이 들어있음

        # test_data fx
        temp_penult = [None] * num_labels
        for label in range(num_labels):
            temp_penult[label] = list()
        for data_num in range(len(test_data)):
            temp_penult[label_test[data_num]].append(test_f_x[data_num])
        penultimate_test_data = np.array(temp_penult)

        # Mu_hat(mean vector for each label, mu_hat[j] = mean of f(x[i]) in label j): calculating mean of the data in each label
        mu_hat = self.get_Mu_hat(penultimate_data, num_neurons, num_labels)

        # Sigma_hat(tied covariance of the distribution of f(x[i]): applying outer-product to all data and calculate the mean --
        sigma_hat = self.get_Sigma_hat(penultimate_data, mu_hat, num_neurons, num_labels, N)

        # Calculate distance of training sample in each label's distribution ---------------------------------------------------
        euc_temp = [None] * num_labels
        mahala_temp = [None] * num_labels
        for label in range(num_labels):
            euc_temp[label] = list()
            mahala_temp[label] = list()

        for label in range(num_labels):
            for datum in penultimate_test_data[label]:
                u = np.reshape(datum, (1, num_neurons))
                v = np.reshape(mu_hat[label], (1, num_neurons))
                euc_temp[label].append(distance.euclidean(u, v, None) ** 2)
        e_dist_data = np.array(euc_temp)  # [0] ~ [9]

        X=np.linalg.inv(sigma_hat)
        for label in range(num_labels):
            for datum in penultimate_test_data[label]:
                u = np.reshape(datum, (1, num_neurons))
                v = np.reshape(mu_hat[label], (1, num_neurons))
                mahala_temp[label].append(distance.mahalanobis(u, v, X) ** 2)
        m_dist_data = np.array(mahala_temp)  # [0] ~ [9]

        m_max = self.Cal_mahalanobis_max(m_dist_data, num_labels)

        return e_dist_data, m_dist_data, m_max

    def Cal_TPR(self, config):
        N_data = len(data)  # 10000
        f_x = np.array(sess.run(fullc2, feed_dict={x: data, keep_prob: 1.0}))
        label_x
        pred_x = np.array(range(N_test))
        for i in range(N_data):
            temp = [None] * num_of_labels
            for label in range(num_of_labels):
                temp[label] = list()
                u = np.reshape(f_x[i], (1, num_of_neurons))
                v = np.reshape(mu_hat[label], (1, num_of_neurons))
                temp[label].append(distance.mahalanobis(u, v, np.linalg.inv(sigma_hat)) ** 2)
            m_dist_data_of_x_test = np.array(temp)
            index = np.argmin(m_dist_data_of_x_test, 0)  # finding index of the closest label
            confidence_score_of_x_test = m_max[index] - m_dist_data_of_x_test[index]  # computing confidence score
            if confidence_score_of_x_test > threshold:
                label_of_x_test[i] = index // 2  # classifying in-distribution data
            else:
                label_of_x_test[i] = ood_index  # classifying out-of-distribution data
        num_of_in_distribution = 0
        num_of_correctly_classified = 0
        accuracy_on_in_distribution = 0.0
        for i in range(N_test):
            if label_of_x_test[i] != ood_index:
                num_of_in_distribution = num_of_in_distribution + 1
                if label_of_x_test[i] == target_label_of_x_test[i]:
                    num_of_correctly_classified = num_of_correctly_classified + 1
        accuracy_on_in_distribution = num_of_correctly_classified / num_of_in_distribution
        tpr = num_of_in_distribution / N_test
        print('Classification accuracy on in-distribution: {:.4f}'.format(accuracy_on_in_distribution))
        print('TPR on in-distribution(MNIST): {:.4f}'.format(tpr), end='\n')

    def Cal_FPR(self, config):
        N_ood = len(X_ood)  # 10000
        X_ood = np.pad(X_ood, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')  # Adding the padding to the dataset
        f_of_x_ood = np.array(sess.run(fullc2, feed_dict={x: X_ood, keep_prob: 1.0}))
        label_of_x_ood = np.array(range(N_ood))
        for i in range(N_ood):
            temp = [None] * num_of_labels
            for label in range(num_of_labels):
                temp[label] = list()
                u = np.reshape(f_of_x_ood[i], (1, num_of_neurons))
                v = np.reshape(mu_hat[label], (1, num_of_neurons))
                temp[label].append(distance.mahalanobis(u, v, np.linalg.inv(sigma_hat)) ** 2)
            m_dist_data_of_x_ood = np.array(temp)
            index = np.argmin(m_dist_data_of_x_ood, 0)  # finding index of the closest label
            confidence_score_of_x_ood = m_max[index] - m_dist_data_of_x_ood[index]  # computing confidence score
            if confidence_score_of_x_ood > threshold:
                label_of_x_ood[i] = index  # classifying in-distribution data
            else:
                label_of_x_ood[i] = ood_index  # classifying out-of-distribution data
        num_of_in_distribution = 0
        for i in range(N_ood):
            if label_of_x_ood[i] != ood_index:
                num_of_in_distribution = num_of_in_distribution + 1
        fpr = num_of_in_distribution / N_ood
        print('FPR on out-of-distribution(EMNIST): {:.4f}'.format(fpr), end='\n')

        '''
        # Set distance threshold for detecting OOD in each label ------------------------------------------------------------
        e_thres_temp = [None] * num_labels
        m_thres_temp = [None] * num_labels
        for label in range(num_labels):
            e_thres_temp[label] = list()
            m_thres_temp[label] = list()
        for label in range(num_labels):
            e_thres_temp[label].append(np.percentile(e_dist_data[label], 95, interpolation='linear'))
            m_thres_temp[label].append(np.percentile(m_dist_data[label], 95, interpolation='linear'))
        e_dist_threshold = np.array(e_thres_temp)
        m_dist_threshold = np.array(m_thres_temp)
        print("<OOD threshold of Mahalanobis distance of penultimate logits in each label>")
        print(m_dist_threshold, end='\n')
        print("<OOD threshold of Euclidean distance of penultimate logits in each label>")
        print(e_dist_threshold, end='\n')
        
        # Show grey image of a random test data and classify it by Mahalanobis classifier --------------------------------------

        # penultimate layer values of test images
        test_f_x = self.sess.run(fullc2, feed_dict={self.images: test_data, self.keep_prob: 1.0})
        
        for random_idx in range(len(test_data)):
            random_data = test_data[random_idx]
            random_data_f_x = test_f_x[random_idx]

            temp = [None] * num_labels
            for label in range(num_labels):
                temp[label] = list()
                u = np.reshape(random_data_f_x, (1, num_neurons))
                v = np.reshape(mu_hat[label], (1, num_neurons))
                temp[label].append(distance.mahalanobis(u, v, np.linalg.inv(sigma_hat)) ** 2)
            dist_to_each_label_mean = np.array(temp)
            print('<Mahalanobis distance of the image from each label\'s mean in penultimate logits>')
            print(dist_to_each_label_mean)
            print()
            idx = np.argmin(dist_to_each_label_mean, 0)  # find the nearest label index
            if dist_to_each_label_mean[idx] >= m_dist_threshold[idx]:  # and check if it is OOD
                print('The image is classified to \'OOD\' by Mahalanobis distance classifier', end='\n')

            else:
                print('The image is classified to \'' + str(int(idx)) + '\' by Mahalanobis distance classifier', end
                      ='\n')
        
        N = len(test_data)
        X_test_y = np.array(range(N))
        X_test_t = test_label
        for i in range(N):
            temp = [None] * num_labels
            for label in range(num_labels):
                temp[label] = list()
                u = np.reshape(test_f_x[i], (1, num_neurons))
                v = np.reshape(mu_hat[label], (1, num_neurons))
                temp[label].append(distance.mahalanobis(u, v, np.linalg.inv(sigma_hat)) ** 2)
            dist_to_each_label_mean = np.array(temp)
            idx = np.argmin(dist_to_each_label_mean, 0)  # find the nearest label index
            X_test_y[i] = idx  # and classify it without considering OOD

        cnt = 0
        acc = 0.0
        for i in range(N):
            if X_test_y[i] == X_test_t[i]:
                cnt = cnt + 1
        acc = cnt / N
        print("마할라노비스분류기:")
        print(acc, end='\n')

        X_test_y = np.array(range(N))
        X_test_t = test_label
        for i in range(N):
            temp = [None] * num_labels
            for label in range(num_labels):
                temp[label] = list()
                u = np.reshape(test_f_x[i], (1, num_neurons))
                v = np.reshape(mu_hat[label], (1, num_neurons))
                temp[label].append(distance.euclidean(u, v, None) ** 2)
            dist_to_each_label_mean = np.array(temp)
            idx = np.argmin(dist_to_each_label_mean, 0)  # find the nearest label index
            X_test_y[i] = idx  # and classify it without considering OOD

        cnt = 0
        acc = 0.0
        for i in range(N):
            if X_test_y[i] == X_test_t[i]:
                cnt = cnt + 1
        acc = cnt / N
        print("유클리디언분류기:")
        print(acc, end='\n')
        '''
