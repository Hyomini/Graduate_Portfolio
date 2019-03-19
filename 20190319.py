import numpy as np
import tensorflow as tf
import os
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.layers import flatten
from sklearn.utils import shuffle
import matplotlib.pylab as plt
from skimage import util
import PIL.Image as pilimg


# Set random seed for permutation
np.random.seed(0)
# Set random seed for tf.Variable initialization
tf.set_random_seed(1234)


# Directory(named 'model') for storing trained model.
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'model')
if os.path.exists(MODEL_DIR) is False:
    os.mkdir(MODEL_DIR)


# LeNet-5 from https://github.com/sujaybabruwad/LeNet-in-Tensorflow/blob/master/LeNet-Lab.ipynb
def lenet(x):
    # Hyperparameters
    mu = 0
    sigma = 0.1
    layer_depth = {'L1': 6, 'L2': 16, 'L3': 120, 'L4': 84, 'L5': 10}

    # Make penultimate logits as global variable
    global fullc2

    # L1: Convolutional ~ (32, 32, 1) -> (28, 28, 6)
    conv1_w = tf.Variable(tf.truncated_normal(shape=[5, 5, 1, layer_depth.get('L1')], mean=mu, stddev=sigma), name='conv1_w')
    conv1_b = tf.Variable(tf.zeros(layer_depth.get('L1')), name='conv1_b')
    conv1 = tf.nn.conv2d(x, conv1_w, strides=[1, 1, 1, 1], padding='VALID') + conv1_b   # 'VALID' - without padding
    conv1 = tf.nn.relu(conv1)
    # Pooling ~ (28, 28, 6) -> (14, 14, 6)
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # L2: Convolutional ~ (14, 14, 6) -> (10, 10, 16)
    conv2_w = tf.Variable(tf.truncated_normal(shape=[5, 5, 6, layer_depth.get('L2')], mean=mu, stddev=sigma), name='conv2_w')
    conv2_b = tf.Variable(tf.zeros(layer_depth.get('L2')), name='conv2_b')
    conv2 = tf.nn.conv2d(pool1, conv2_w, strides=[1, 1, 1, 1], padding='VALID') + conv2_b
    conv2 = tf.nn.relu(conv2)
    # Pooling ~ (10, 10, 16) -> (5, 5, 16)
    pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # Flatten ~ (5, 5, 16) -> (400)
    fullc1 = flatten(pool2)
    # L3: Fully-connected ~ (400) -> (120)
    fullc1_w = tf.Variable(tf.truncated_normal(shape=(400, layer_depth.get('L3')), mean=mu, stddev=sigma), name='fullc1_w')
    fullc1_b = tf.Variable(tf.zeros(layer_depth.get('L3')), name='fullc1_b')
    fullc1 = tf.matmul(fullc1, fullc1_w) + fullc1_b
    fullc1 = tf.nn.relu(fullc1)

    # L4: Fully-connected ~ (120) -> (84)
    fullc2_w = tf.Variable(tf.truncated_normal(shape=(120, layer_depth.get('L4')), mean=mu, stddev=sigma), name='fullc2_w')
    fullc2_b = tf.Variable(tf.zeros(layer_depth.get('L4')), name='fullc2_b')
    fullc2 = tf.matmul(fullc1, fullc2_w) + fullc2_b
    fullc2 = tf.nn.relu(fullc2)

    # L5: Fully-connected ~ (84) -> (10)
    fullc3_w = tf.Variable(tf.truncated_normal(shape=(layer_depth.get('L4'), layer_depth.get('L5')), mean=mu, stddev=sigma), name='fullc3_w')
    fullc3_b = tf.Variable(tf.zeros(layer_depth.get('L5')), name='fullc3_b')
    # logits
    y = tf.matmul(fullc2, fullc3_w) + fullc3_b
    return y


def loss(y, t):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=t)
    loss_function = tf.reduce_mean(cross_entropy)
    return loss_function


def train_step(loss):
    optimizer = tf.train.GradientDescentOptimizer(0.01)
    train_step = optimizer.minimize(loss)
    return train_step


def accuracy(y, t):
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(t, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return accuracy


if __name__ == '__main__':
    # Get data==========================================================================================================
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
    # get permuted N train data
    n = len(X_train)
    N = 20000
    indices = np.random.permutation(range(n))[:N]
    X_train = X_train[indices]
    Y_train = Y_train[indices]
    # Pad images(70000, 28, 28, 1) with 0s - pad windows on both sides by 2 but not for the datasets and channels
    X_train = np.pad(X_train, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')    # Constant_values's default is 0
    X_validation = np.pad(X_validation, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')
    X_test = np.pad(X_test, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')

    # Set model ========================================================================================================
    x = tf.placeholder(tf.float32, shape=[None, 32, 32, 1])
    t = tf.placeholder(tf.int32, shape=[None])
    one_hot_t = tf.one_hot(t, 10)   # One_hot representation for target values
    y = lenet(x)
    loss = loss(y, one_hot_t)
    train_step = train_step(loss)
    accuracy = accuracy(y, one_hot_t)

    # Train and evaluate model =========================================================================================
    '''
    # 1.Store(Train) ---------------------------------------------------------------------------------------------------
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    sess = tf.Session()
    sess.run(init)
    epochs = 20
    batch_size = 200
    n_batches = N // batch_size
    for epoch in range(epochs):
        X_, Y_ = shuffle(X_train, Y_train)
        for i in range(n_batches):
            start = i * batch_size
            end = start + batch_size
            sess.run(train_step, feed_dict={x: X_[start:end], t: Y_[start:end]})
        val_loss = loss.eval(session=sess, feed_dict={x: X_, t: Y_})
        val_acc = accuracy.eval(session=sess, feed_dict={x: X_, t: Y_})
        print('epoch:', epoch+1, ' loss:', val_loss, ' accuracy:', val_acc)
    # Evaluate accuracy of validation and test datasets
    val_acc_v = accuracy.eval(session=sess, feed_dict={x: X_validation, t: Y_validation})
    val_acc_t = accuracy.eval(session=sess, feed_dict={x: X_test, t: Y_test})
    print()
    print('validation accuracy: {:.4f}'.format(val_acc_v))
    print('test accuracy:       {:.4f}'.format(val_acc_t), end='\n')
    # Store model
    model_path = saver.save(sess, MODEL_DIR + './model.ckpt')
    print('Model saved to:', model_path)
    print()
    # '''
    # '''
    # 2.Restore --------------------------------------------------------------------------------------------------------
    saver = tf.train.Saver()
    sess = tf.Session()
    saver.restore(sess, MODEL_DIR + '/model.ckpt')
    # Evaluate accuracy of validation and test datasets
    val_acc_v = accuracy.eval(session=sess, feed_dict={x: X_validation, t: Y_validation})
    val_acc_t = accuracy.eval(session=sess, feed_dict={x: X_test, t: Y_test})
    print()
    print('validation accuracy: {:.4f}'.format(val_acc_v))
    print('test accuracy:       {:.4f}'.format(val_acc_t), end='\n')
    # '''

    # (N, epochs) = (20000, 20) 일 때, (validation accuracy, test accuracy) = (0.9550, 0.9565)
    # ...

# generative classifier under Eculidian distance -> GDA
# A list for f(x): [[], [], [], [], [], [], [], [], [], []]
a = [None]*10
for i in range(10):
    a[i] = list()

# Get f(x) and append to 'a' distinguished by label
label = sess.run(tf.argmax(y, 1), feed_dict={x: X_train})
f_x = sess.run(fullc2, feed_dict={x: X_train})  #  이거 접근범위 에러-> 로짓 글로벌 설정 or 모델 함수 x

for i in range(len(label)):
    a[label[i]].append(f_x[i])

data = np.array(a)

a = [None]*10
for i in range(10):
    a[i] = list()
for i in range(10):
    if len(data[i]) == 0:
        a[i] = [0]
    elif len(data[i]) == 1:
        a[i] = data[i][0]
    else:
        for j in range(len(data[i])-2):
            data[i][0] += data[i][j+1]  # Sum all data and put it in the first position of each label
        a[i] = data[i][0] / len(data[i])

mean = np.array(a)
# print(mean)

# Distance of training datasets
a = [None]*10
for i in range(10):
    a[i] = list()

for i in range(len(label)):
    a[label[i]].append(sum((f_x[i] - mean[label[i]]) ** 2))
# 여기서 공분산 np로 구해서 라이브러리 입력하면될듯?
distance = np.array(a)

# 정규분포 라이브러리-> P(f(x)| y=c) = N(f(x)|mu, sigma-hat) 계산값 argmax 하는 classifier 씨발 ****************************

# print(distance[i][int(len(data)*(95.0/100.0))-1])    # 95퍼센트 위치에 해당하는 인덱스의 값(근사)
# print(np.percentile(data, 95))                       # 95% 백분위수 출력(데이터의 95%가 발견되는 기댓값)
a = [None]*10
for i in range(10):
    a[i] = list()
    a[i].append(np.percentile(distance[i], 95))

ood_distance = np.array(a)
print("<ODD threshold of Euclidian distance of penultimate logits in each label>")
print(ood_distance)
print()

# 각 클래스별 디스턴스 분포를 정규화시킨 히스토그램으로 표현
for i in range(10):
    data = np.sort(distance[i])
    bins = np.arange(0, 1000, 10)  # 도수분포구간 설정
    plt.hist(data, bins, normed=True)
    plt.title("Class: %d" % i)
    plt.xlabel('Distance', fontsize=15)
    plt.ylabel('Num of data', fontsize=15)
    plt.show(block=True)

# 무작위 데이터 선택
index = 2
img = mnist.test.images[index]
'''
# 직접 그린 OOD 선택
ood_img = pilimg.open("./mldata/ood0.bmp")
ood_img = 1 - np.array(ood_img)/255
ood_img = np.reshape(ood_img, [1, 784])
img = ood_img
'''

# 가우시안 노이즈(원리 모름)
gd_noised_img = util.random_noise(img, mode='gaussian', clip=True)
# 가우시안 노이즈에 이미지 각 지점에 국소 분산 추가(?)
lv_noised_img = util.random_noise(img, mode='localvar', clip=True)
# 포아숑 분포 노이즈 ??
pd_noised_img = util.random_noise(img, mode='poisson', clip=True)
# Salt: 무작위 픽셀을 1로 대체
st_noised_img = util.random_noise(img, mode='salt', clip=True)
# Pepper: 무작위 픽셀을 0으로 대체
pp_noised_img = util.random_noise(img, mode='pepper', clip=True)
# S&P: Salt or Pepper
sp_noised_img = util.random_noise(img, mode='s&p', clip=True)
# Speckle: 반점(image += n * image, n is uniform noise with specified mean & variance)
sc_noised_img = util.random_noise(img, mode='speckle', clip=True)

# 노이즈 옵션 선택
img = sp_noised_img

'''
# 픽셀 하나하나 0~1사이 소수점으로 표현
for index, pixel in enumerate(img):
    if (index % 28) == 0:
        print('\n')
    else:
        print("%5f" % pixel, end="")
print('\n')
'''

# 그레이 이미지로 표현
plt.figure(figsize=(5, 5))
plt.imshow(np.reshape(img, [28, 28]), cmap='Greys')
plt.show()

# Euclidian distance classifier(?) ~ 가장 가까운 distance를 가지는 클래스로 분류하고 ood_distance와 비교
img_f_x = sess.run(fullc2, feed_dict={x: X_test})

a = [None]*10
for i in range(10):
    a[i] = list()
    a[i].append(sum((img_f_x[index] - mean[i]) ** 2))

class_distance = np.array(a)
print('<Euclidian distance of the image from each label\'s mean in penultimate logits>')
print(class_distance)
print()

print("===============================================================================================================")
i = np.argmin(class_distance, 0)
if class_distance[i] >= ood_distance[i]:
    print('The image is classified to \'OOD\' by Euclidian distance classifier', end='\n')
else:
    print('The image is classified to \'' + str(int(i)) + '\' by Euclidian distance classifier', end='\n')
print("===============================================================================================================")
