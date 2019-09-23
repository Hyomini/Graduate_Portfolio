import numpy as np
import tensorflow as tf
import os
import time
from tensorflow.examples.tutorials.mnist import input_data
from scipy.spatial import distance
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import seaborn as sns
import gzip as gz
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from six.moves import cPickle as pickle
from skimage import util


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
k = 2  # the number of K means

# Data(data[j][k]: (k+1)th f(x) of label j): distinguishing f(x[i]) by its label ---------------------------------------
temp = [None] * num_of_labels
for label in range(num_of_labels):
    temp[label] = list()                                                      # making temporary list to use list.append
for i in range(N):
    temp[label_of_x[i]].append(f_of_x[i])
data = np.array(temp)                                             # changing the list to numpy array for numpy operation

temp = [None] * k * num_of_labels
for label in range(k * num_of_labels):
    temp[label] = list()
for label in range(num_of_labels):
    t_sne_data = TSNE(learning_rate=100).fit_transform(data[label])                    # dimension reduction using t-SNE
    data_frame = pd.DataFrame(t_sne_data, columns=('x', 'y'))            # making 2-dimensional data frame by t-SNE data
    data_points = data_frame.values
    k_means_data = KMeans(n_clusters=k).fit(data_points)                                            # k-means clustering
    data_frame['cluster_id'] = k_means_data.labels_                              # labeling the cluster id of data frame
    #sns.lmplot('x', 'y', data=data_frame, fit_reg=False, scatter_kws={"s": 500}, hue="cluster_id")
    #plt.title('K-mean plot')
    #plt.xlabel('x')
    #plt.ylabel('y')
    #plt.show()
    data_frame = np.array(data_frame)
    for i in range(len(data[label])):
        temp[k * label + int(data_frame[i][2])].append(data[label][i])
data2 = np.array(temp)
num_of_labels_split = k * num_of_labels
data_split = data2

# Mu_hat(mean vector for each label, mu_hat[j] = mean of f(x[i]) in label j): calculating mean of the data in each label
mu_hat = [np.zeros(num_of_neurons)] * num_of_labels
for label in range(num_of_labels):
    mu_hat[label] = np.mean(data[label], axis=0)

mu_hat_split = [np.zeros(num_of_neurons)] * num_of_labels_split
for label in range(num_of_labels_split):
    mu_hat_split[label] = np.mean(data_split[label], axis=0)

# Sigma_hat(tied covariance of the distribution of f(x[i]): applying outer-product to all data and calculate the mean --
sigma_hat = np.zeros([num_of_neurons, num_of_neurons])
for label in range(num_of_labels):
    for datum in data[label]:
        u = np.reshape(datum, (num_of_neurons, 1))
        v = np.reshape(mu_hat[label], (num_of_neurons, 1))
        sigma_hat += np.matmul((u - v), (u - v).T)
sigma_hat_split = np.zeros([num_of_neurons, num_of_neurons])
for label in range(num_of_labels_split):
    for datum in data_split[label]:
        u = np.reshape(datum, (num_of_neurons, 1))
        v = np.reshape(mu_hat_split[label], (num_of_neurons, 1))
        sigma_hat_split += np.matmul((u - v), (u - v).T)

sigma_hat = sigma_hat / N
sigma_hat_split = sigma_hat_split / N

linalg = np.linalg.inv(sigma_hat)
linalg_split = np.linalg.inv(sigma_hat_split)

# M(E)_dist_data(m(e)_dist_data[j][k]: (k+1)th Mahalanobis(Euclidean) distance data of label j) ------------------------
temp = [None] * num_of_labels
for label in range(num_of_labels):
    temp[label] = list()
for label in range(num_of_labels):
    for datum in data[label]:
        u = np.reshape(datum, (1, num_of_neurons))
        v = np.reshape(mu_hat[label], (1, num_of_neurons))
        # temp[label].append(distance.euclidean(u, v, None) ** 2)
        temp[label].append(distance.mahalanobis(u, v, linalg) ** 2)
# e_dist_data = np.array(temp)
m_dist_data = np.array(temp)

temp = [None] * num_of_labels_split
for label in range(num_of_labels_split):
    temp[label] = list()
for label in range(num_of_labels_split):
    for datum in data_split[label]:
        u = np.reshape(datum, (1, num_of_neurons))
        v = np.reshape(mu_hat_split[label], (1, num_of_neurons))
        # temp[label].append(distance.euclidean(u, v, None) ** 2)
        temp[label].append(distance.mahalanobis(u, v, linalg_split) ** 2)
# e_dist_data_split = np.array(temp)
m_dist_data_split = np.array(temp)

# Max(distance threshold, max_dist_data[j] = Mahalanobis(Euclidean) distance threshold of label j) ---------------------
temp = [None] * num_of_labels
for label in range(num_of_labels):
    temp[label] = list()
for label in range(num_of_labels):
    # temp[label].append(np.percentile(e_dist_data[label], 95, interpolation='linear'))
    temp[label].append(np.percentile(m_dist_data[label], 95, interpolation='linear'))
max_dist_data = np.array(temp)

temp = [None] * num_of_labels_split
for label in range(num_of_labels_split):
    temp[label] = list()
for label in range(num_of_labels_split):
    # temp[label].append(np.percentile(e_dist_data_split[label], 95, interpolation='linear'))
    temp[label].append(np.percentile(m_dist_data_split[label], 95, interpolation='linear'))
max_dist_data_split = np.array(temp)

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
# Receiver operating characteristic curve ==============================================================================
########################################################################################################################
threshold = -2.8                       # threshold variable for Mahalanobis(Euclidean) distance-based confidence score
ood_index = -1                                                             # giving label -1 to out-of-distribution data

fp = open("0918M_log.txt",'w')

roc_x_em = []
roc_x_em_split = []
roc_x_notm = []
roc_x_notm_split = []
roc_y = []
roc_y_split = []

# Mnist
N_test = len(X_test)                                                                                             # 10000
f_of_x_test = np.array(sess.run(fullc2, feed_dict={x: X_test, keep_prob: 1.0}))
target_label_of_x_test = Y_test

# EMnist
f = gz.open('EMNIST_data/emnist-letters-train-images-idx3-ubyte.gz', 'r')  # EMNIST dataset for out-of-distribution data
f.read(16)
buf = f.read(28 * 28 * 10000)
emnist_test = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
emnist_test = emnist_test.reshape([10000, 28, 28, 1])
emnist_test = emnist_test / 255.0                                                # scaling from 0.0 ~ 255.0 to 0.0 ~ 1.0
X_ood1 = emnist_test
N_ood1 = len(X_ood1)                                                                                             # 10000
'''
# Calibration techniques -----------------------------------------------------------------------------------------------
for i in range(N_ood):
    for j in range(3):
        X_ood1[i] = util.random_noise(X_ood1[i], mode='pepper', clip=True)                 # Adding the noise to dataset
'''

X_ood1 = np.pad(X_ood1, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')                 # Adding the padding to the dataset
f_of_x_ood1 = np.array(sess.run(fullc2, feed_dict={x: X_ood1, keep_prob: 1.0}))

# NOTMnist
with open('NOTMNIST_data/notMNIST.pickle', 'rb') as file:                # NOTMNIST dataset for out-of-distribution data
    data_list = []
    while True:
        try:
            data = pickle.load(file)
        except EOFError:
            break
        data_list.append(data)
notmnist_test = data_list[0]['test_dataset']
notmnist_test = notmnist_test.reshape([10000, 28, 28, 1])
notmnist_test = notmnist_test + 0.5                                               # scaling from -0.5 ~ 0.5 to 0.0 ~ 1.0
X_ood2 = notmnist_test
N_ood2 = len(X_ood2)                                                                                             # 10000

for i in range(N_ood2):
    for j in range(3):
        X_ood2[i] = util.random_noise(X_ood2[i], mode='pepper', clip=True)                  # Adding the noise to dataset

X_ood2 = np.pad(X_ood2, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')                # Adding the padding to the dataset
f_of_x_ood2 = np.array(sess.run(fullc2, feed_dict={x: X_ood2, keep_prob: 1.0}))



start_time = time.time()
for threshold in range(-150, 150, 10):
    # TPR on in-distribution(MNIST test dataset) ---------------------------------------------------------------------------
    label_of_x_test = np.array(range(N_test))
    for i in range(N_test):
        temp = [None] * num_of_labels
        for label in range(num_of_labels):
            temp[label] = list()
            u = np.reshape(f_of_x_test[i], (1, num_of_neurons))
            v = np.reshape(mu_hat[label], (1, num_of_neurons))
            # temp[label].append(distance.euclidean(u, v, None) ** 2)
            temp[label].append(distance.mahalanobis(u, v, linalg) ** 2)
        dist_data_of_x_test = np.array(temp)
        index = np.argmin(dist_data_of_x_test, 0)  # finding index of the closest label
        confidence_score_of_x_test = max_dist_data[index] - dist_data_of_x_test[index]  # computing confidence score
        if confidence_score_of_x_test > threshold:
            label_of_x_test[i] = index  # classifying as in-distribution data
        else:
            label_of_x_test[i] = ood_index  # classifying as out-of-distribution data
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
    roc_y.append(tpr)
    print('Classification accuracy on in-distribution: {:.4f}'.format(accuracy_on_in_distribution))
    print('TPR on in-distribution(MNIST): {:.4f}'.format(tpr), end='\n')

    # TPR on in-distribution(MNIST test dataset, split) ---------------------------------------------------------------------------
    label_of_x_test = np.array(range(N_test))
    for i in range(N_test):
        temp = [None] * num_of_labels_split
        for label in range(num_of_labels_split):
            temp[label] = list()
            u = np.reshape(f_of_x_test[i], (1, num_of_neurons))
            v = np.reshape(mu_hat_split[label], (1, num_of_neurons))
            # temp[label].append(distance.euclidean(u, v, None) ** 2)
            temp[label].append(distance.mahalanobis(u, v, linalg_split) ** 2)
        dist_data_of_x_test = np.array(temp)
        index = np.argmin(dist_data_of_x_test, 0)  # finding index of the closest label
        confidence_score_of_x_test = max_dist_data_split[index] - dist_data_of_x_test[index]  # computing confidence score
        if confidence_score_of_x_test > threshold:
            label_of_x_test[i] = index // k  # classifying as in-distribution data
        else:
            label_of_x_test[i] = ood_index  # classifying as out-of-distribution data
    num_of_in_distribution = 0
    num_of_correctly_classified = 0
    accuracy_on_in_distribution = 0.0
    for i in range(N_test):
        if label_of_x_test[i] != ood_index:
            num_of_in_distribution = num_of_in_distribution + 1
            if label_of_x_test[i] == target_label_of_x_test[i]:
                num_of_correctly_classified = num_of_correctly_classified + 1
    accuracy_on_in_distribution = num_of_correctly_classified / num_of_in_distribution
    tpr_split = num_of_in_distribution / N_test
    roc_y_split.append(tpr_split)

    # FPR on out-of-distribution 1(EMNIST dataset) -------------------------------------------------------------------------
    label_of_x_ood1 = np.array(range(N_ood1))
    for i in range(N_ood1):
        temp = [None] * num_of_labels
        for label in range(num_of_labels):
            temp[label] = list()
            u = np.reshape(f_of_x_ood1[i], (1, num_of_neurons))
            v = np.reshape(mu_hat[label], (1, num_of_neurons))
            # temp[label].append(distance.euclidean(u, v, None) ** 2)
            temp[label].append(distance.mahalanobis(u, v, linalg) ** 2)
        dist_data_of_x_ood1 = np.array(temp)
        index = np.argmin(dist_data_of_x_ood1, 0)  # finding index of the closest label
        confidence_score_of_x_ood1 = max_dist_data[index] - dist_data_of_x_ood1[index]  # computing confidence score
        if confidence_score_of_x_ood1 > threshold:
            label_of_x_ood1[i] = index  # classifying as in-distribution data
        else:
            label_of_x_ood1[i] = ood_index  # classifying as out-of-distribution data
    num_of_in_distribution1 = 0
    for i in range(N_ood1):
        if label_of_x_ood1[i] != ood_index:
            num_of_in_distribution1 = num_of_in_distribution1 + 1
    fpr1 = num_of_in_distribution1 / N_ood1
    roc_x_em.append(fpr1)
    print('FPR on out-of-distribution(ENMIST): {:.4f}'.format(fpr1), end='\n')

    # FPR on out-of-distribution 1(EMNIST dataset, split) -------------------------------------------------------------------------
    label_of_x_ood1 = np.array(range(N_ood1))
    for i in range(N_ood1):
        temp = [None] * num_of_labels_split
        for label in range(num_of_labels_split):
            temp[label] = list()
            u = np.reshape(f_of_x_ood1[i], (1, num_of_neurons))
            v = np.reshape(mu_hat_split[label], (1, num_of_neurons))
            # temp[label].append(distance.euclidean(u, v, None) ** 2)
            temp[label].append(distance.mahalanobis(u, v, linalg_split) ** 2)
        dist_data_of_x_ood1 = np.array(temp)
        index = np.argmin(dist_data_of_x_ood1, 0)  # finding index of the closest label
        confidence_score_of_x_ood1 = max_dist_data_split[index] - dist_data_of_x_ood1[index]  # computing confidence score
        if confidence_score_of_x_ood1 > threshold:
            label_of_x_ood1[i] = index // k # classifying as in-distribution data
        else:
            label_of_x_ood1[i] = ood_index  # classifying as out-of-distribution data
    num_of_in_distribution1 = 0
    for i in range(N_ood1):
        if label_of_x_ood1[i] != ood_index:
            num_of_in_distribution1 = num_of_in_distribution1 + 1
    fpr1_split = num_of_in_distribution1 / N_ood1
    roc_x_em_split.append(fpr1_split)

    # FPR on out-of-distribution 2(NOTMNIST dataset) -----------------------------------------------------------------------
    label_of_x_ood2 = np.array(range(N_ood2))
    for i in range(N_ood2):
        temp = [None] * num_of_labels
        for label in range(num_of_labels):
            temp[label] = list()
            u = np.reshape(f_of_x_ood2[i], (1, num_of_neurons))
            v = np.reshape(mu_hat[label], (1, num_of_neurons))
            # temp[label].append(distance.euclidean(u, v, None) ** 2)
            temp[label].append(distance.mahalanobis(u, v, linalg) ** 2)
        dist_data_of_x_ood2 = np.array(temp)
        index = np.argmin(dist_data_of_x_ood2, 0)  # finding index of the closest label
        confidence_score_of_x_ood2 = max_dist_data[index] - dist_data_of_x_ood2[index]  # computing confidence score
        if confidence_score_of_x_ood2 > threshold:
            label_of_x_ood2[i] = index  # classifying as in-distribution data
        else:
            label_of_x_ood2[i] = ood_index  # classifying as out-of-distribution data
    num_of_in_distribution2 = 0
    for i in range(N_ood2):
        if label_of_x_ood2[i] != ood_index:
            num_of_in_distribution2 = num_of_in_distribution2 + 1
    fpr2 = num_of_in_distribution2 / N_ood2
    roc_x_notm.append(fpr2)
    print('FPR on out-of-distribution(NOTMNIST): {:.4f}'.format(fpr2), end='\n')

    # FPR on out-of-distribution 1(NOTMNIST dataset, split) -------------------------------------------------------------------------
    label_of_x_ood2 = np.array(range(N_ood2))
    for i in range(N_ood2):
        temp = [None] * num_of_labels_split
        for label in range(num_of_labels_split):
            temp[label] = list()
            u = np.reshape(f_of_x_ood2[i], (1, num_of_neurons))
            v = np.reshape(mu_hat_split[label], (1, num_of_neurons))
            # temp[label].append(distance.euclidean(u, v, None) ** 2)
            temp[label].append(distance.mahalanobis(u, v, linalg_split) ** 2)
        dist_data_of_x_ood2 = np.array(temp)
        index = np.argmin(dist_data_of_x_ood2, 0)  # finding index of the closest label
        confidence_score_of_x_ood2 = max_dist_data_split[index] - dist_data_of_x_ood2[index]  # computing confidence score
        if confidence_score_of_x_ood2 > threshold:
            label_of_x_ood2[i] = index // k # classifying as in-distribution data
        else:
            label_of_x_ood2[i] = ood_index  # classifying as out-of-distribution data
    num_of_in_distribution2 = 0
    for i in range(N_ood2):
        if label_of_x_ood2[i] != ood_index:
            num_of_in_distribution2 = num_of_in_distribution2 + 1
    fpr2_split = num_of_in_distribution2 / N_ood2
    roc_x_notm_split.append(fpr2_split)
    repeat_time = time.time()
    print('Time: {:.2f}s'.format(repeat_time - start_time))
    print('Threshold: {}\n'.format(threshold))

plt.figure()
plt.plot(roc_x_em, roc_y, color='red', label='emnist')
plt.plot(roc_x_notm, roc_y, color='blue', label='notmnist')
plt.plot(roc_x_em_split, roc_y_split, color='green', label='emnist(split)')
plt.plot(roc_x_notm_split, roc_y_split, color='black', label='notmnist(split)')
plt.title("ROC curve")
plt.xticks([0, 0.5, 1])
plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
plt.xlabel("FPR on out-of-distribution")
plt.ylabel("TPR on in-distribution (mnist)")
plt.xlim([0,1])
plt.ylim([0,1])
plt.legend()
plt.show()
fp.close()





# Euclidean classifier ROC curve data ----------------------------------------------------------------------------------
# (threshold, TPR_MNIST, FPR_EMNIST, FPR-NOTMNIST) = (-(INF), 1.0000, 1.0000, 1.0000)
# (threshold, TPR_MNIST, FPR_EMNIST, FPR-NOTMNIST) = (-150.0, 0.9998, 0.9976, 0.9909)
# (threshold, TPR-MNIST, FPR-EMNIST, FPR-NOTMNIST) = ( -50.0, 0.9864, 0.7149, 0.5090)
# (threshold, TPR-MNIST, FPR-EMNIST, FPR-NOTMNIST) = (  -4.3, 0.9500, 0.3661, 0.2219)
# (threshold, TPR-MNIST, FPR-EMNIST, FPR-NOTMNIST) = (  17.4, 0.9000, 0.2313, 0.1227)
# (threshold, TPR-MNIST, FPR-EMNIST, FPR-NOTMNIST) = (  29.0, 0.8500, 0.1720,       )
# (threshold, TPR-MNIST, FPR-EMNIST, FPR-NOTMNIST) = (  36.5, 0.8000, 0.1392,       )
# (threshold, TPR-MNIST, FPR_EMNIST, FPR-NOTMNIST) = (  41.7, 0.7500, 0.1201,       )
# (threshold, TPR-MNIST, FPR_EMNIST, FPR-NOTMNIST) = (  ... , 0. ..., 0. ..., 0. ...)
# (threshold, TPR-MNIST, FPR_EMNIST, FPR-NOTMNIST) = ( 150.0, 0.0233, 0.0004, 0.0001)
# (threshold, TPR-MNIST, FPR-EMNIST, FPR_NOTMNIST) = ( 170.0, 0.0021, 0.0000, 0.0000)
# (threshold, TPR-MNIST, FPR_EMNIST, FPR_NOTMNIST) = ( (INF), 0.0000, 0.0000, 0.0000)

# Mahalanobis classifier ROC curve data 1 ------------------------------------------------------------------------------
# (threshold, TPR_MNIST, FPR_EMNIST, FPR-NOTMNIST) = (-(INF), 1.0000, 1.0000, 1.0000)
# (threshold, TPR-MNIST, FPR_EMNIST, FPR-NOTMNIST) = (  ... , 0. ..., 0. ..., 0. ...)
# (threshold, TPR_MNIST, FPR_EMNIST, FPR-NOTMNIST) = ( (INF), 0.0000, 0.0000, 0.0000)

# Mahalanobis classifier ROC curve data 2(Calibration technique - Input pre-processing) --------------------------------
# (threshold, TPR_MNIST, FPR_EMNIST, FPR-NOTMNIST) = (-(INF), 1.0000, 1.0000, 1.0000)
# (threshold, TPR-MNIST, FPR_EMNIST, FPR-NOTMNIST) = (  ... , 0. ..., 0. ..., 0. ...)
# (threshold, TPR_MNIST, FPR_EMNIST, FPR-NOTMNIST) = ( (INF), 0.0000, 0.0000, 0.0000)