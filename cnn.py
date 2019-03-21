import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.utils import shuffle
import matplotlib.pylab as plt
from skimage import util
import PIL.Image as pilimg


np.random.seed(0)
tf.set_random_seed(1234)

mnist = datasets.fetch_mldata("MNIST original")

n = len(mnist.data)
N = 10000
train_size = 0.8
indices = np.random.permutation(range(n))[:N]

X = mnist.data[indices]
y = mnist.target[indices]
Y = np.eye(10)[y.astype(int)]

print(X,Y)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=train_size)

n_in = len(X[0])  # 784
n_hidden = 200
n_out = len(Y[0])  # 10

x = tf.placeholder(tf.float32, shape=[None, n_in])
x_img = tf.reshape(x, [-1,28,28,1])
t = tf.placeholder(tf.float32, shape=[None, n_out])
keep_prob = tf.placeholder(tf.float32)

W0 = tf.Variable(tf.truncated_normal([3,3,1,32], stddev=0.01))  # 표준편자 0.1로 하면 GVP
L0 = tf.nn.conv2d(x_img, W0, strides = [1,1,1,1], padding = 'SAME')
L0 = tf.nn.relu(L0)
L0 = tf.nn.max_pool(L0, ksize=[1,2,2,1], strides=[1,2,2,1], padding = 'SAME')
L0 = tf.nn.dropout(L0, keep_prob=0.8)

W1 = tf.Variable(tf.truncated_normal([3,3,32,64], stddev=0.01))
L1 = tf.nn.conv2d(L0,W1,strides=[1,1,1,1],padding='SAME')
L1 = tf.nn.relu(L1)
L1 = tf.nn.max_pool(L1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
L1 = tf.nn.dropout(L1, keep_prob=0.8)

W2 = tf.Variable(tf.truncated_normal([3,3,64,128], stddev=0.01))
L2 = tf.nn.conv2d(L1, W2, strides=[1,1,1,1], padding='SAME')
L2 = tf.nn.relu(L2)
L2 = tf.nn.max_pool(L2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
L2 = tf.nn.dropout(L2, keep_prob = 0.8)
L2 = tf.reshape(L2, [-1,128*4*4])

W3 = tf.get_variable("W3", shape=[128*4*4,625],initializer=tf.contrib.layers.xavier_initializer())
b3 = tf.Variable(tf.truncated_normal([625]))
h2 = tf.nn.relu(tf.matmul(L2,W3) + b3)
h2 = tf.nn.dropout(h2, keep_prob=0.8)

W4 = tf.get_variable("W4", shape=[625,10], initializer = tf.contrib.layers.xavier_initializer())
b4 = tf.Variable(tf.truncated_normal([10]))
y = tf.nn.softmax(tf.matmul(h2,W4) + b4)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(t * tf.log(y), axis=1))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(t, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

epochs = 20
batch_size = 200

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

n_batches = (int)(N * train_size) // batch_size

for epoch in range(epochs):
    X_, Y_ = shuffle(X_train, Y_train)

    for i in range(n_batches):
        start = i * batch_size
        end = start + batch_size

        sess.run(train_step, feed_dict={x: X_[start:end], t: Y_[start:end]})

    loss = cross_entropy.eval(session=sess, feed_dict={x: X_, t: Y_})
    acc = accuracy.eval(session=sess, feed_dict={x: X_, t: Y_})
    print('epoch:', epoch+1, ' loss:', loss, ' accuracy:', acc)

accuracy_rate = accuracy.eval(session=sess, feed_dict={x: X_test, t: Y_test})

print('test accuracy: ', accuracy_rate, end='\n')
print()


# A list for f(x): [[], [], [], [], [], [], [], [], [], []]
tmp = [None]*10
for i in range(10):
    tmp[i] = list()

# Get f(x) and append to 'a' distinguished by label
label = sess.run(tf.argmax(y, 1), feed_dict={x: X_train})
f_x = sess.run(h2, feed_dict={x: X_train})

for i in range(len(label)):
    tmp[label[i]].append(np.array(f_x[i]))

data = np.array(tmp)
tmp = np.array(tmp)

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
            data[i][0] += data[i][j+1]  # 각 클래스별 첫번째 열에 데이터의 합 입력
        a[i] = data[i][0] / len(data[i])

mean = np.array(a)

"""
a = [None]*10
for i in range(10):
    a[i] = list()

for i in range(10):
    if len(tmp[i]) == 0:
        a[i] = [0]
    elif len(tmp[i]) == 1:
        a[i] = tmp[i][0]
    else:
        for j in range(len(tmp[i])-2):
            tmp[i][0] += (tmp[i][j+1] - mean[i])@(tmp[i][j+1]-mean[i]).T
        a[i] = tmp[i][0]

for i in range(9):
    a[0] += a[i+1]

a[0] = a[0] / (train_size * 10000) 
"""




# distance of training datasets
a = [None]*10
for i in range(10):
    a[i] = list()

for i in range(len(label)):
    a[label[i]].append(sum((f_x[i] - mean[label[i]]) ** 2))

distance = np.array(a)

# print(distance[i][int(len(data)*(95.0/100.0))-1])  # 95퍼센트 위치에 해당하는 인덱스의 값(근사)
# print(np.percentile(data, 95))  # 95% 백분위수 출력(데이터의 95%가 발견되는 기댓값)
a = [None]*10
for i in range(10):
    a[i] = list()
    a[i].append(np.percentile(distance[i], 95))

ood_distance = np.array(a)
print("<각 클래스별 ODD distance>")
print(ood_distance)
print()

# 각 클래스별 디스턴스 분포를 정규화시킨 히스토그램으로 표현
for i in range(10):
    data = np.sort(distance[i])
    bins = np.arange(0, 2000, 20)  # 도수분포구간 설정
    plt.hist(data, bins, normed=True)
    plt.title("Class: %d" % i)
    plt.xlabel('Distance', fontsize=15)
    plt.ylabel('Num of data', fontsize=15)
    plt.show(block=True)

# 무작위 데이터 선택
index = 10
img = mnist.data[indices][index]
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
g_img = np.reshape(img, [28, 28])
plt.imshow(g_img, cmap='Greys')
plt.show()

# GDA classifier ~ 가장 가까운 distance를 가지는 클래스로 분류하고 ood_distance와 비교
img_f_x = sess.run(h2, feed_dict={x: mnist.data[indices]})

a = [None]*10
for i in range(10):
    a[i] = list()
    a[i].append(sum((img_f_x[index] - mean[i]) ** 2))

class_distance = np.array(a)
print('<해당 이미지의 각 클래스별 distance>')
print(class_distance)
print()

i = np.argmin(class_distance, 0)
if class_distance[i] >= ood_distance[i]:
    print('해당 이미지는 GDA에 의해 \"OOD\"로 분류')
else:
    print('해당 이미지는 GDA에 의해 \"' + str(int(i)) + '\" 로 분류')


'''
iod = list()    # in-distribution
ood = list()    # out-of-distribution
num = [0]*10

for temp in range(test_size):
    test_fx = np.array((L2.eval(session=sess, feed_dict={
        X: mnist.test.images[temp:temp+1]}))[0])
    sum1 = 100000
    index = 0
    for x in range(10):
        sum_now = sum((data_avg[x] - test_fx) ** 2)
        if sum1 > sum_now:
            sum1 = sum_now
            index = x
    if sum1 > 300:  # 거리가 130 이상이면 OOD
        OOD.append(temp)
    else:
        IOD.append(temp)
    num[index] = num[index] + 1

print("\nNumber of OOD: ", len(OOD))
print("OOD Percentage: ", len(OOD)/test_size*100, "%")
for x in range(10):
    print(x, ":", num[x])
'''
