# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tarfile
from IPython.display import display, Image
from scipy import ndimage
from sklearn.linear_model import LogisticRegression
from six.moves.urllib.request import urlretrieve
from six.moves import cPickle as pickle

url = 'http://commondatastorage.googleapis.com/books1000/'
last_percent_reported = None

import numpy as np
import tensorflow as tf
import os
from tensorflow.examples.tutorials.mnist import input_data
from scipy.spatial import distance
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import seaborn as sns
import gzip as gz
from sklearn.utils import shuffle
import matplotlib.pyplot as plt


# Giving specific random seed for data permutation and tf.Variable initialization
np.random.seed(0)
tf.set_random_seed(1234)


# Directory(named 'model') for storing trained model
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'model')
if os.path.exists(MODEL_DIR) is False:
    os.mkdir(MODEL_DIR)


# LeNet-5(https://github.com/sujaybabruwad/LeNet-in-Tensorflow/blob/master/LeNet-Lab.ipynb) ~ 'L3':120->128, 'L4':84->64
def lenet(x):
    # Hyperparameters
    mu = 0
    sigma = 0.1
    layer_depth = {'L1': 6, 'L2': 16, 'L3': 128, 'L4': 64, 'L5': 10}

    # Making the variable to global one for generative classifier
    global conv1, conv2, fullc1, fullc2

    # L1: Convolutional ~ (32, 32, 1) -> (28, 28, 6)
    conv1_w = tf.Variable(np.sqrt(2.0/32*32)*tf.truncated_normal(shape=[5, 5, 1, layer_depth.get('L1')], mean=mu, stddev=sigma), name='conv1_w')
    conv1_b = tf.Variable(tf.zeros(layer_depth.get('L1')), name='conv1_b')
    conv1 = tf.nn.conv2d(x, conv1_w, strides=[1, 1, 1, 1], padding='VALID') + conv1_b
    conv1 = tf.nn.relu(conv1)
    conv1 = tf.nn.dropout(conv1, keep_prob)
    # Pooling ~ (28, 28, 6) -> (14, 14, 6)
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # L2: Convolutional ~ (14, 14, 6) -> (10, 10, 16)
    conv2_w = tf.Variable(np.sqrt(2.0/14*14)*tf.truncated_normal(shape=[5, 5, layer_depth.get('L1'), layer_depth.get('L2')], mean=mu, stddev=sigma), name='conv2_w')
    conv2_b = tf.Variable(tf.zeros(layer_depth.get('L2')), name='conv2_b')
    conv2 = tf.nn.conv2d(pool1, conv2_w, strides=[1, 1, 1, 1], padding='VALID') + conv2_b
    conv2 = tf.nn.relu(conv2)
    conv2 = tf.nn.dropout(conv2, keep_prob)
    # Pooling ~ (10, 10, 16) -> (5, 5, 16)
    pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # Flatten ~ (5, 5, 16) -> (400)
    fullc1 = tf.contrib.layers.flatten(pool2)

    # L3: Fully-connected ~ (400) -> (128)
    fullc1_w = tf.Variable(np.sqrt(2.0/400)*tf.truncated_normal(shape=(400, layer_depth.get('L3')), mean=mu, stddev=sigma), name='fullc1_w')
    fullc1_b = tf.Variable(tf.zeros(layer_depth.get('L3')), name='fullc1_b')
    fullc1 = tf.matmul(fullc1, fullc1_w) + fullc1_b
    fullc1 = tf.nn.relu(fullc1)
    fullc1 = tf.nn.dropout(fullc1, keep_prob)

    # L4: Fully-connected ~ (128) -> (64)
    fullc2_w = tf.Variable(np.sqrt(2.0/120)*tf.truncated_normal(shape=(layer_depth.get('L3'), layer_depth.get('L4')), mean=mu, stddev=sigma), name='fullc2_w')
    fullc2_b = tf.Variable(tf.zeros(layer_depth.get('L4')), name='fullc2_b')
    fullc2 = tf.matmul(fullc1, fullc2_w) + fullc2_b
    fullc2 = tf.nn.relu(fullc2)
    fullc2 = tf.nn.dropout(fullc2, keep_prob)

    # L5: Fully-connected ~ (64) -> (10)
    fullc3_w = tf.Variable(tf.truncated_normal(shape=(layer_depth.get('L4'), layer_depth.get('L5')), mean=mu, stddev=sigma), name='fullc3_w')
    fullc3_b = tf.Variable(tf.zeros(layer_depth.get('L5')), name='fullc3_b')
    logits = tf.matmul(fullc2, fullc3_w) + fullc3_b                                                             # output

    return logits


def loss(y, t):
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=t))


def train_step(loss):
    return tf.train.AdamOptimizer(1e-3).minimize(loss)


def accuracy(y, t):
    return tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y, 1), tf.argmax(t, 1)), tf.float32))


if __name__ == '__main__':
    # Getting data =====================================================================================================
    mnist = input_data.read_data_sets("MNIST_data/", reshape=False)
    X_train, Y_train = mnist.train.images, mnist.train.labels
    X_validation, Y_validation = mnist.validation.images, mnist.validation.labels
    X_test, Y_test = mnist.test.images, mnist.test.labels
    '''
    print("Image Shape: {}".format(X_train[0].shape))        # (28, 28, 1)
    print("Image Shape: {}".format(Y_train[0].shape))                 # ()
    print("Training Set:   {} samples".format(len(X_train)))       # 55000
    print("Validation Set: {} samples".format(len(X_validation)))   # 5000
    print("Test Set:       {} samples".format(len(X_test)))        # 10000
    '''
    # Getting random N training data
    n = len(X_train)
    N = 30000
    indices = np.random.permutation(range(n))[:N]
    X_train = X_train[indices]
    Y_train = Y_train[indices]
    # Adding the padding to the dataset
    X_train = np.pad(X_train, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')                       # constant(default): 0
    X_validation = np.pad(X_validation, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')
    X_test = np.pad(X_test, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')

    # Seting model =====================================================================================================
    x = tf.placeholder(tf.float32, shape=[None, 32, 32, 1])
    t = tf.placeholder(tf.int32, shape=[None])
    one_hot_t = tf.one_hot(t, 10)
    keep_prob = tf.placeholder(tf.float32)
    y = lenet(x)
    loss = loss(y, one_hot_t)
    train_step = train_step(loss)
    accuracy = accuracy(y, one_hot_t)

    # Training and evaluating model ====================================================================================
    '''
    # 1.Store(Train) ---------------------------------------------------------------------------------------------------   
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    sess = tf.Session()
    sess.run(init)
    epochs = 25
    batch_size = 200
    n_batches = N // batch_size
    for epoch in range(epochs):
        X_, Y_ = shuffle(X_train, Y_train)
        for i in range(n_batches):
            start = i * batch_size
            end = start + batch_size
            sess.run(train_step, feed_dict={x: X_[start:end], t: Y_[start:end], keep_prob: 0.8})
        val_loss = loss.eval(session=sess, feed_dict={x: X_, t: Y_, keep_prob: 1.0})
        val_acc = accuracy.eval(session=sess, feed_dict={x: X_, t: Y_, keep_prob: 1.0})
        print('epoch:', epoch+1, ' loss:', val_loss, ' accuracy:', val_acc)
    val_acc_v = accuracy.eval(session=sess, feed_dict={x: X_validation, t: Y_validation, keep_prob: 1.0})
    val_acc_t = accuracy.eval(session=sess, feed_dict={x: X_test, t: Y_test, keep_prob: 1.0})
    print()
    print('validation accuracy: {:.4f}'.format(val_acc_v))
    print('test accuracy:       {:.4f}'.format(val_acc_t))
    print()
    model_path = saver.save(sess, MODEL_DIR + './model.ckpt')
    print('Model saved to:', model_path)
    print()
    '''
    # 2.Restore --------------------------------------------------------------------------------------------------------
    saver = tf.train.Saver()
    sess = tf.Session()
    saver.restore(sess, MODEL_DIR + '/model.ckpt')
    val_acc_v = accuracy.eval(session=sess, feed_dict={x: X_validation, t: Y_validation, keep_prob: 1.0})
    val_acc_t = accuracy.eval(session=sess, feed_dict={x: X_test, t: Y_test, keep_prob: 1.0})
    print()
    print('validation accuracy: {:.4f}'.format(val_acc_v))
    print('test accuracy:       {:.4f}'.format(val_acc_t))
    print()
    # when (epochs, batch_size, keep_drop) is (25, 200, 0.8), (validation accuracy, test accuracy) is (0.9892, 0.9891)


########################################################################################################################
# Generative classifier ================================================================================================
########################################################################################################################
N = len(X_train)                                                              # the number of training data(x[i]): 30000
f_of_x = np.array(sess.run(fullc2, feed_dict={x: X_train, keep_prob: 1.0}))   # the output of the layer of x[i]: f(x[i])
label_of_x = np.array(sess.run(tf.argmax(y, 1), feed_dict={x: X_train, keep_prob: 1.0}))      # classified label of x[i]
num_of_labels = 10                                                                         # number of labels: 10(MNIST)
num_of_neurons = len(f_of_x[0])                                         # number of neurons(dimensions) in the layer: 64
k = 2                                                                                       # k in k-means clustering: 2

# Data(data[j][k]: (k+1)th f(x) of label j): distinguishing f(x[i]) by its label ---------------------------------------
temp = [None] * num_of_labels
for label in range(num_of_labels):
    temp[label] = list()                                                      # making temporary list to use list.append
for i in range(N):
    temp[label_of_x[i]].append(f_of_x[i])
data = np.array(temp)                                             # changing the list to numpy array for numpy operation

# Data2(data2[j][k]: (k+1)th f(x) of label j): spliting label and arrange data -----------------------------------------
temp = [None] * k * num_of_labels
for label in range(k * num_of_labels):
    temp[label] = list()
for label in range(num_of_labels):
    t_sne_data = TSNE(learning_rate=100).fit_transform(data[label])                    # dimension reduction using t-SNE
    data_frame = pd.DataFrame(t_sne_data, columns=('x', 'y'))            # making 2-dimensional data frame by t-SNE data
    data_points = data_frame.values
    k_means_data = KMeans(n_clusters=k).fit(data_points)                                            # k-means clustering
    data_frame['cluster_id'] = k_means_data.labels_                              # labeling the cluster id of data frame
    sns.lmplot('x', 'y', data=data_frame, fit_reg=False, scatter_kws={"s": 500}, hue="cluster_id")
    plt.title('K-mean plot')
    plt.xlabel('x')
    plt.ylabel('y')
    # plt.show()
    data_frame = np.array(data_frame)
    for i in range(len(data[label])):
        temp[k * label + int(data_frame[i][2])].append(data[label][i])
data2 = np.array(temp)
num_of_labels = k * num_of_labels
data_split = data2

# Mu_hat(mean vector for each label, mu_hat[j] = mean of f(x[i]) in label j): calculating mean of the data in each label
mu_hat = [np.zeros(num_of_neurons)] * num_of_labels
for label in range(num_of_labels):
    mu_hat[label] = np.mean(data[label], axis=0)

# Sigma_hat(tied covariance of the distribution of f(x[i]): applying outer-product to all data and calculate the mean --
sigma_hat = np.zeros([num_of_neurons, num_of_neurons])
for label in range(num_of_labels):
    for datum in data[label]:
        u = np.reshape(datum, (num_of_neurons, 1))
        v = np.reshape(mu_hat[label], (num_of_neurons, 1))
        sigma_hat += np.matmul((u - v), (u - v).T)
sigma_hat = sigma_hat / N

# M(E)_dist_data(m(e)_dist_data[j][k]: (k+1)th Mahalanobis(Euclidean) distance data of label j) ------------------------
temp = [None] * num_of_labels
for label in range(num_of_labels):
    temp[label] = list()
for label in range(num_of_labels):
    for datum in data[label]:
        u = np.reshape(datum, (1, num_of_neurons))
        v = np.reshape(mu_hat[label], (1, num_of_neurons))
        # temp[label].append(distance.euclidean(u, v, None) ** 2)
        temp[label].append(distance.mahalanobis(u, v, np.linalg.inv(sigma_hat)) ** 2)
# e_dist_data = np.array(temp)
m_dist_data = np.array(temp)

# Max(distance threshold, max_dist_data[j] = Mahalanobis(Eucliean) distance threshold of label j) ----------------------
temp = [None] * num_of_labels
for label in range(num_of_labels):
    temp[label] = list()
for label in range(num_of_labels):
    # temp[label].append(np.percentile(e_dist_data[label], 95, interpolation='linear'))
    temp[label].append(np.percentile(m_dist_data[label], 95, interpolation='linear'))
max_dist_data = np.array(temp)

# Histogram for each label's Mahalanobis(Euclidean) distance distribution ----------------------------------------------
'''
for label in range(num_of_labels):
    # data = np.sort(e_dist_data[label])
    data = np.sort(m_dist_data[label])
    bins = np.arange(0, 300, 2)
    plt.hist(data, bins, normed=True)
    plt.title("label: %d" % label)
    plt.xlabel('distance', fontsize=15)
    plt.ylabel('num of data', fontsize=15)
    plt.show(block=True)
'''


########################################################################################################################
# Receiver operating characteristic point ==============================================================================
########################################################################################################################
threshold = 80                         # threshold variable for Mahalanobis(Euclidean) distance-based confidence score
ood_index = -1                                                             # giving label -1 to out-of-distribution data
f = gz.open('EMNIST_data/emnist-letters-train-images-idx3-ubyte.gz', 'r')  # EMNIST dataset for out-of-distribution data
f.read(16)
buf = f.read(28 * 28 * 10000)
X_ood = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
X_ood = X_ood.reshape(10000, 28, 28, 1)
X_ood = X_ood / 255.0                                                      # scaling X_ood from 0.0 ~ 255.0 to 0.0 ~ 1.0
X_ood = np.pad(X_ood, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')  # Adding the padding to the dataset
fp = open("EMNIST_B_M_log.txt",'w')

for threshold in range(150):
    # TPR on in-distribution(MNIST test dataset) ---------------------------------------------------------------------------
    N_test = len(X_test)  # 10000
    f_of_x_test = np.array(sess.run(fullc2, feed_dict={x: X_test, keep_prob: 1.0}))
    target_label_of_x_test = Y_test
    label_of_x_test = np.array(range(N_test))
    for i in range(N_test):
        temp = [None] * num_of_labels
        for label in range(num_of_labels):
            temp[label] = list()
            u = np.reshape(f_of_x_test[i], (1, num_of_neurons))
            v = np.reshape(mu_hat[label], (1, num_of_neurons))
            # temp[label].append(distance.euclidean(u, v, None) ** 2)
            temp[label].append(distance.mahalanobis(u, v, np.linalg.inv(sigma_hat)) ** 2)
        dist_data_of_x_test = np.array(temp)
        index = np.argmin(dist_data_of_x_test, 0)  # finding index of the closest label
        confidence_score_of_x_test = max_dist_data[index] - dist_data_of_x_test[index]  # computing confidence score
        if confidence_score_of_x_test > threshold:
            label_of_x_test[i] = index // k  # classifying in-distribution data
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
    write_data = 'Classification accuracy on in-distribution: {:.4f}\n'.format(accuracy_on_in_distribution)
    fp.write(write_data)
    write_data = 'TPR on in-distribution(MNIST): {:.4f}\n'.format(tpr)
    fp.write(write_data)

    # FPR on out-of-distribution(EMNIST dataset) ---------------------------------------------------------------------------
    N_ood = len(X_ood)  # 10000
    f_of_x_ood = np.array(sess.run(fullc2, feed_dict={x: X_ood, keep_prob: 1.0}))
    label_of_x_ood = np.array(range(N_ood))
    for i in range(N_ood):
        temp = [None] * num_of_labels
        for label in range(num_of_labels):
            temp[label] = list()
            u = np.reshape(f_of_x_ood[i], (1, num_of_neurons))
            v = np.reshape(mu_hat[label], (1, num_of_neurons))
            # temp[label].append(distance.euclidean(u, v, None) ** 2)
            temp[label].append(distance.mahalanobis(u, v, np.linalg.inv(sigma_hat)) ** 2)
        dist_data_of_x_ood = np.array(temp)
        index = np.argmin(dist_data_of_x_ood, 0)  # finding index of the closest label
        confidence_score_of_x_ood = max_dist_data[index] - dist_data_of_x_ood[index]  # computing confidence score
        if confidence_score_of_x_ood > threshold:
            label_of_x_ood[i] = index  # classifying in-distribution data
        else:
            label_of_x_ood[i] = ood_index  # classifying out-of-distribution data
    num_of_in_distribution = 0
    for i in range(N_ood):
        if label_of_x_ood[i] != ood_index:
            num_of_in_distribution = num_of_in_distribution + 1
    fpr = num_of_in_distribution / N_ood
    print('FPR on out-of-distribution(ENMIST): {:.4f}'.format(fpr), end='\n')
    write_data = 'FPR on out-of-distribution(EMNIST): {:.4f}\n'.format(fpr)
    fp.write(write_data)
fp.close()

# (distance, threshold, TPR, FPR) = (E , , , ,)
# (distance, threshold, TPR, FPR) = (M , -2.8, 0.9499, 0.4657 )