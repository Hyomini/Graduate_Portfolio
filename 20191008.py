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
from six.moves import cPickle as pickle
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

# Data1(data1[j][k]: (k+1)th f(x) of label j): distinguishing f(x[i]) by its label -------------------------------------
temp = [None] * num_of_labels
for label in range(num_of_labels):
    temp[label] = list()                                                      # making temporary list to use list.append
for i in range(N):
    temp[label_of_x[i]].append(f_of_x[i])
data1 = np.array(temp)                                            # changing the list to numpy array for numpy operation

# Data2(data2[j][k]: (k+1)th f(x) of label j): spliting label and arrange data using k-means clustering ----------------
temp = [None] * k * num_of_labels
for label in range(k * num_of_labels):
    temp[label] = list()
for label in range(num_of_labels):
    k_means_data = KMeans(n_clusters=k).fit(data1[label])                                           # k-means clustering
    print(k_means_data.labels_)
    for i in range(len(data1[label])):
        temp[k * label + int(k_means_data.labels_[i])].append(data1[label][i])
data2 = np.array(temp)
num_of_labels = k * num_of_labels
data = data2
'''
# Data2: spliting label and arrange data using t-sne and k-means clustering --------------------------------------------
temp = [None] * k * num_of_labels
for label in range(k * num_of_labels):
    temp[label] = list()
for label in range(num_of_labels):
    t_sne_data = TSNE(learning_rate=100).fit_transform(data1[label])                   # dimension reduction using t-SNE
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
    for i in range(len(data1[label])):
        temp[k * label + int(data_frame[i][2])].append(data1[label][i])
data2 = np.array(temp)
num_of_labels = k * num_of_labels
data = data2
'''

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
inv_sigma_hat = np.linalg.inv(sigma_hat)

# M(E)_dist_data(m(e)_dist_data[j][k]: (k+1)th Mahalanobis(Euclidean) distance data of label j) ------------------------
temp = [None] * num_of_labels
for label in range(num_of_labels):
    temp[label] = list()
for label in range(num_of_labels):
    for datum in data[label]:
        u = np.reshape(datum, (1, num_of_neurons))
        v = np.reshape(mu_hat[label], (1, num_of_neurons))
        # temp[label].append(distance.euclidean(u, v, None) ** 2)
        temp[label].append(distance.mahalanobis(u, v, inv_sigma_hat) ** 2)
# e_dist_data = np.array(temp)
m_dist_data = np.array(temp)

# Max(distance threshold, max_dist_data[j] = Mahalanobis(Eucliean) distance threshold of label j) ----------------------
temp = [None] * num_of_labels
for label in range(num_of_labels):
    temp[label] = list()
for label in range(num_of_labels):
    # temp[label].append(np.percentile(e_dist_data[label], 95, interpolation='linear'))
#sqeqwefwqefewfqwefqwefewqefqew#
    temp[label].append(np.percentile(m_dist_data[label], 80, interpolation='linear'))
#qwerweqrqwer#
max_dist_data = np.array(temp)
print(max_dist_data)

# Improve covariance ###################################################################################################
# 단순히 OOD로 분류되는 (트레이닝)데이터들을 제외한 값으로 다시 공분산(sigma_hat)계산 ##########################################
# Calculate distance of training sample in each label's distribution ---------------------------------------------------
temp = [None] * num_of_labels
for label in range(num_of_labels):
    temp[label] = list()
for label in range(num_of_labels):
    for datum in data[label]:
        u = np.reshape(datum, (1, num_of_neurons))
        v = np.reshape(mu_hat[label], (1, num_of_neurons))
        # p[label].append(distance.euclidean(u, v, None) ** 2)
        temp[label].append(distance.mahalanobis(u, v, inv_sigma_hat) ** 2)
# Delete the f_x_data if it is OOD -------------------------------------------------------------------------------------
data = list(data)  # f_x_data
N = 0
for label in range(num_of_labels):
    for i in range(len(temp[label])):
        if temp[label][i] > max_dist_data[label]:
            data[label][i] = [-1] * num_of_neurons  # OOD 는 뉴런이 다 -1이게하고
    for iterate in range(3000):
        for i in range(len(data[label])):
            isOOD = 1
            for j in range(num_of_neurons):
                if data[label][i][j] != -1:
                    isOOD = 0
                    break
            if isOOD == 1:
                data[label].pop(i)
                break
    N += len(data[label])
print(N)
data = np.array(data)
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
inv_sigma_hat = np.linalg.inv(sigma_hat)

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
threshold = -30.3                         # threshold variable for Mahalanobis(Euclidean) distance-based confidence score
ood_index = -1                                                             # giving label -1 to out-of-distribution data

# TPR on in-distribution(MNIST test dataset) ---------------------------------------------------------------------------
N_test = len(X_test)                                                                                             # 10000
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
        temp[label].append(distance.mahalanobis(u, v, inv_sigma_hat) ** 2)
    dist_data_of_x_test = np.array(temp)
    index = np.argmin(dist_data_of_x_test, 0)                                       # finding index of the closest label
    confidence_score_of_x_test = max_dist_data[index] - dist_data_of_x_test[index]          # computing confidence score
    if confidence_score_of_x_test > threshold:
        label_of_x_test[i] = index // k                                               # classifying in-distribution data
    else:
        label_of_x_test[i] = ood_index                                            # classifying out-of-distribution data
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

# FPR on out-of-distribution 1(EMNIST dataset) -------------------------------------------------------------------------
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
f_of_x_ood = np.array(sess.run(fullc2, feed_dict={x: X_ood1, keep_prob: 1.0}))
label_of_x_ood = np.array(range(N_ood1))
for i in range(N_ood1):
    temp = [None] * num_of_labels
    for label in range(num_of_labels):
        temp[label] = list()
        u = np.reshape(f_of_x_ood[i], (1, num_of_neurons))
        v = np.reshape(mu_hat[label], (1, num_of_neurons))
        # temp[label].append(distance.euclidean(u, v, None) ** 2)
        temp[label].append(distance.mahalanobis(u, v, inv_sigma_hat)**2)
    dist_data_of_x_ood = np.array(temp)
    index = np.argmin(dist_data_of_x_ood, 0)                                        # finding index of the closest label
    confidence_score_of_x_ood = max_dist_data[index] - dist_data_of_x_ood[index]            # computing confidence score
    if confidence_score_of_x_ood > threshold:
        label_of_x_ood[i] = index // k                                                # classifying in-distribution data
    else:
        label_of_x_ood[i] = ood_index                                             # classifying out-of-distribution data
num_of_in_distribution = 0
for i in range(N_ood1):
    if label_of_x_ood[i] != ood_index:
        num_of_in_distribution = num_of_in_distribution + 1
fpr = num_of_in_distribution / N_ood1
print('FPR on out-of-distribution(ENMIST): {:.4f}'.format(fpr), end='\n')

# FPR on out-of-distribution 2(NOTMNIST dataset) -----------------------------------------------------------------------
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
'''
for i in range(N_ood):
    for j in range(3):
        X_ood2[i] = util.random_noise(X_ood[i], mode='pepper', clip=True)                  # Adding the noise to dataset
'''
X_ood2 = np.pad(X_ood2, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')                # Adding the padding to the dataset
f_of_x_ood = np.array(sess.run(fullc2, feed_dict={x: X_ood2, keep_prob: 1.0}))
label_of_x_ood = np.array(range(N_ood2))
for i in range(N_ood2):
    temp = [None] * num_of_labels
    for label in range(num_of_labels):
        temp[label] = list()
        u = np.reshape(f_of_x_ood[i], (1, num_of_neurons))
        v = np.reshape(mu_hat[label], (1, num_of_neurons))
        # temp[label].append(distance.euclidean(u, v, None) ** 2)
        temp[label].append(distance.mahalanobis(u, v, inv_sigma_hat)**2)
    dist_data_of_x_ood = np.array(temp)
    index = np.argmin(dist_data_of_x_ood, 0)                                        # finding index of the closest label
    confidence_score_of_x_ood = max_dist_data[index] - dist_data_of_x_ood[index]            # computing confidence score
    if confidence_score_of_x_ood > threshold:
        label_of_x_ood[i] = index // k                                                # classifying in-distribution data
    else:
        label_of_x_ood[i] = ood_index                                             # classifying out-of-distribution data
num_of_in_distribution = 0
for i in range(N_ood2):
    if label_of_x_ood[i] != ood_index:
        num_of_in_distribution = num_of_in_distribution + 1
fpr = num_of_in_distribution / N_ood2
print('FPR on out-of-distribution(NOTMNIST): {:.4f}'.format(fpr), end='\n')

# Euclidean classifier ROC curve data ----------------------------------------------------------------------------------
# (threshold, TPR_MNIST, FPR_EMNIST, FPR-NOTMNIST) = (-(INF), 1.0000, 1.0000, 1.0000)
# (threshold, TPR_MNIST, FPR_EMNIST, FPR-NOTMNIST) = (-150.0, 0.9998, 0.9976, 0.9909)
# (threshold, TPR-MNIST, FPR-EMNIST, FPR-NOTMNIST) = ( -50.0, 0.9864, 0.7149, 0.5090)
# (threshold, TPR-MNIST, FPR-EMNIST, FPR-NOTMNIST) = (  -4.3, 0.9500, 0.3661, 0.2219)
# (threshold, TPR-MNIST, FPR-EMNIST, FPR-NOTMNIST) = (  17.4, 0.9000, 0.2313, 0.1227)
### (threshold, TPR-MNIST, FPR-EMNIST, FPR-NOTMNIST) = (  29.0, 0.8500, 0.1720,       )
# (threshold, TPR-MNIST, FPR-EMNIST, FPR-NOTMNIST) = (  36.5, 0.8000, 0.1392,       )
# (threshold, TPR-MNIST, FPR_EMNIST, FPR-NOTMNIST) = (  41.7, 0.7500, 0.1201,       )
# (threshold, TPR-MNIST, FPR_EMNIST, FPR-NOTMNIST) = (  ... , 0. ..., 0. ..., 0. ...)
# (threshold, TPR-MNIST, FPR_EMNIST, FPR-NOTMNIST) = ( 150.0, 0.0233, 0.0004, 0.0001)
# (threshold, TPR-MNIST, FPR-EMNIST, FPR_NOTMNIST) = ( 170.0, 0.0021, 0.0000, 0.0000)
# (threshold, TPR-MNIST, FPR_EMNIST, FPR_NOTMNIST) = ( (INF), 0.0000, 0.0000, 0.0000)

# Mahalanobis classifier ROC curve data 1 ------------------------------------------------------------------------------
# (threshold, TPR_MNIST, FPR_EMNIST, FPR-NOTMNIST) = (-(INF), 1.0000, 1.0000, 1.0000)
# (threshold. TPR_MNIST, FPR_EMNIST, FPR_NOTMNIST) = (-200.0, 0.9990, 0.9386, 0.7185)
# (threshold. TPR_MNIST, FPR_EMNIST, FPR_NOTMNIST) = ( -70.0, 0.9909, 0.7291, 0.4414)
# (threshold. TPR_MNIST, FPR_EMNIST, FPR_NOTMNIST) = (  -2.8, 0.9500, 0.4140, 0.2399)
# (threshold. TPR_MNIST, FPR_EMNIST, FPR_NOTMNIST) = (  23.5, 0.9000, 0.2530, 0.1221)
### (threshold. TPR_MNIST, FPR_EMNIST, FPR_NOTMNIST) = (  37.7, 0.8500, 0.1823, 0.0650)
# (threshold, TPR-MNIST, FPR_EMNIST, FPR-NOTMNIST) = (  ... , 0. ..., 0. ..., 0. ...)
# (threshold, TPR_MNIST, FPR_EMNIST, FPR-NOTMNIST) = ( (INF), 0.0000, 0.0000, 0.0000)

# Mahalanobis classifier ROC curve data 2(Using improved inverse matirx of covaraince: percent:90)----------------------
# percentile (60~ 95)
# (threshold, TPR_MNIST, FPR_EMNIST, FPR-NOTMNIST) = (-(INF), 1.0000, 1.0000, 1.0000)
# (threshold. TPR_MNIST, FPR_EMNIST, FPR_NOTMNIST) = (  )
# (threshold. TPR_MNIST, FPR_EMNIST, FPR_NOTMNIST) = ( -20.3, 0.8766, 0.2039, 0.1171)
### (threshold. TPR_MNIST, FPR_EMNIST, FPR_NOTMNIST) = (  -9.3, 0.8500, 0.1619, 0.0781)
# (threshold. TPR_MNIST, FPR_EMNIST, FPR_NOTMNIST) = (  )
# (threshold, TPR-MNIST, FPR_EMNIST, FPR-NOTMNIST) = (  ... , 0. ..., 0. ..., 0. ...)
# (threshold, TPR_MNIST, FPR_EMNIST, FPR-NOTMNIST) = ( (INF), 0.0000, 0.0000, 0.0000)

# Mahalanobis classifier ROC curve data 3(당연히 공분산 개선하고 스플릿까지  ) ---------------------------------------------
# percentile (40 ~ 80)
# (threshold, TPR_MNIST, FPR_EMNIST, FPR-NOTMNIST) = (-(INF), 1.0000, 1.0000, 1.0000)
# (threshold. TPR_MNIST, FPR_EMNIST, FPR_NOTMNIST) = (      )
### (threshold. TPR_MNIST, FPR_EMNIST, FPR_NOTMNIST) = (  -9.3, 0.8500, 0.1630,   )
# (threshold. TPR_MNIST, FPR_EMNIST, FPR_NOTMNIST) = (  )
# (threshold, TPR-MNIST, FPR_EMNIST, FPR-NOTMNIST) = (  ... , 0. ..., 0. ..., 0. ...)
# (threshold, TPR_MNIST, FPR_EMNIST, FPR-NOTMNIST) = ( (INF), 0.0000, 0.0000, 0.0000)
