import tensorflow as tf
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pylab as plt
import os
from tensorflow.examples.tutorials.mnist import input_data
from scipy.spatial import distance

# Set random seed for permutation
np.random.seed(0)

# Set random seed for tf.Variable initialization
tf.set_random_seed(1234)
 
# Directory(named 'model') for storing trained model.
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'model')
if os.path.exists(MODEL_DIR) is False:
    os.mkdir(MODEL_DIR)


# Loading Mnist datasets
mnist = input_data.read_data_sets("./mnist/data/", one_hot=True)
 
# Making Placeholder and model ------------------------------------------------
def CNN(x):
    
    depth = {'C1':16, 'C2': 32, 'C3': 64, 'F1' : 256, 'Class':10}
    
    global h_fc1
     
    W_conv1 = tf.Variable(tf.truncated_normal(shape=[5, 5, 1, depth.get('C1')], stddev=5e-2))
    b_conv1 = tf.Variable(tf.constant(0.1, shape=[depth.get('C1')]))
    h_conv1 = tf.nn.relu(tf.nn.conv2d(x, W_conv1, strides=[1, 1, 1, 1], padding='SAME') + b_conv1)
    h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
     
    # 28x28x16 -> 14x14x32
    W_conv2 = tf.Variable(tf.truncated_normal(shape=[5,5,depth.get('C1'),depth.get('C2')], stddev=5e-2))
    b_conv2 = tf.Variable(tf.constant(0.1, shape=[depth.get('C2')]))
    h_conv2 = tf.nn.relu(tf.nn.conv2d(h_pool1,W_conv2, strides=[1,1,1,1], padding='SAME') + b_conv2)
    h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1,3,3,1], strides=[1,2,2,1], padding='SAME')
     
    # 14x14x32 -> 7x7x64
    W_conv3 = tf.Variable(tf.truncated_normal(shape=[5,5,depth.get('C2'),depth.get('C3')], stddev=5e-2))
    b_conv3 = tf.Variable(tf.constant(0.1, shape=[depth.get('C3')]))
    h_conv3 = tf.nn.relu(tf.nn.conv2d(h_pool2,W_conv3, strides=[1,1,1,1], padding='SAME') + b_conv3)
     
    # 7x7x64 -> 3136
    h_conv3 = tf.reshape(h_conv3, [-1, 3136])
     
    # 3136 -> 256
    W_fc1 = tf.Variable(tf.truncated_normal(shape=[3136,depth.get('F1')]))
    b_fc1 = tf.Variable(tf.constant(0.1, shape=[depth.get('F1')]))
    h_fc1 = tf.matmul(h_conv3,W_fc1) + b_fc1
     
    # 256 -> 10
    W_fc2 = tf.Variable(tf.truncated_normal(shape=[depth.get('F1'),depth.get('Class')]))
    b_fc2 = tf.Variable(tf.constant(0.1, shape=[depth.get('Class')]))
     
    logits = tf.matmul(h_fc1,W_fc2) + b_fc2
    
    return logits

def loss(logits, labels):
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels))

def train_step(loss):
    return tf.train.AdamOptimizer(0.001).minimize(loss)

def accuracy(logits, labels):
    return tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1)), tf.float32))
  
    
# Set Placeholder and model ----------------------------------------------------
X = tf.placeholder(tf.float32, [None, 784])
X_img = tf.reshape(X, [-1, 28,28,1])
Y = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)

y = CNN(X_img)
loss = loss(y,Y)
train_step = train_step(loss)

pred = tf.nn.softmax(y)
accuracy = accuracy(pred,Y)


# Training -------------------------------------------------------------------- 
init = tf.global_variables_initializer()
sess = tf.Session()
saver = tf.train.Saver()
sess.run(init)
 
batch_size = 128
total_batch = int(mnist.train.num_examples / batch_size)
 
for epoch in range(10):
    total_cost = 0
 
    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        
        _, cost_val = sess.run([train_step,loss], feed_dict={X: batch_xs, Y:batch_ys, keep_prob : 0.5})
        total_cost += cost_val
        
    print('Epoch :', '%04d' % (epoch +1), 'Avg. cost =', '{:.3f}'.format(total_cost / total_batch))
    
print('정확도', sess.run(accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels,keep_prob : 1.0}))
 

# Saving ----------------------------------------------------------------------
model_path = saver.save(sess, MODEL_DIR)
print('Model saved to:', model_path)
print()
 

 

""" 
# Restoring -------------------------------------------------------------------
#tf.reset_default_graph()
saver = tf.train.Saver()
sess = tf.Session()
saver.restore(sess, 'C:/Users/lg/Desktop/model') # 경로 후 파일 이름 -> checkpoint 있는 폴더 안에서 meta,index,data 앞의 이름만 적어줘야함 
                                                   # ex. 위 path는 바탕화면에 model.meta, model.index, model.data가 있을 경우임  
print('정확도', sess.run(accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels,keep_prob : 1.0}))
"""
 
# Arrange Penultimate outputs -------------------------------------------------
classified_output = [None]*10
for i in range(10):
    classified_output[i] = list()
 
# Get f(x) and append to 'a' distinguished by label ---------------------------
label,penul_output = sess.run([tf.argmax(pred, 1),h_fc1], feed_dict={X: mnist.train.images})
 
for i in range(len(label)):
    classified_output[label[i]].append(np.array(penul_output[i]))
 
classified_output = np.array(classified_output)    
data = classified_output
output_sum = np.zeros(256)

# Calculating Mean of each class ----------------------------------------------
mean = [None] * 10
for i in range(10):
    mean[i] = list()

for i in range(10):
    if len(classified_output[i]) == 0:
        mean[i] = [0]
    elif len(classified_output[i]) == 1:
        mean[i] = classified_output[i][0]
    else:
        for element in range(len(classified_output[i])):
            output_sum += classified_output[i][element]
        mean[i] = output_sum / len(classified_output[i])
        output_sum = np.zeros(256)
 
# Calculating Covariance of whole classes -------------------------------------
covar = np.zeros((256,256))
 
for i_class in range(10):
    for sample in classified_output[i]:
        v = np.reshape(sample , (256,1))
        u = np.reshape(mean[i_class], (256,1))
        covar += np.matmul((v-u),(v-u).T)
        
covar = covar/len(mnist.train.images) # default : (256,256)   ※공분산 -> 모든 샘플이 linear 관계라면 행렬식 0, 
                                            # 따라서 한 샘플 정돈 어긋날 거라고 가정하고 예외처리 안함
 

# Calculating Mahalanobis distance between Training sample and predicted classification distribution
Mdist = [None]*10
for i in range(10):
    Mdist[i] = list()
 
for i in range(len(label)):
    v = np.reshape(penul_output[i], (1,256)) #벡터로 transpose 안되는 경우 있음
    u = np.reshape(mean[label[i]], (1,256))
    Mdist[label[i]].append(distance.mahalanobis(v,u,np.linalg.inv(covar))**2)
 
Mdist = np.array(Mdist)
 
# Choosing threshold -> default : 95% 
threshold = [None]*10
for i in range(10):
    threshold[i] = list()
    threshold[i].append(np.percentile(Mdist[i], 95))
 
ood_distance = np.array(threshold)
print("<각 클래스별 OOD Mahalanobis distance>")
print(ood_distance)
print()


""" Not Completed  !!! --------------------------------------------------------

# Calculating Mahalanobis distance between Test sample and predicted classification distribution
# Decide whether this distance is OOD or not.
label,penul_output = 0
label, penul_output = sess.run([tf.argmax(pred, 1),h_fc1], feed_dict={X: mnist.test.images})

Mdist = [None]

for i in range(len(label)):
    v = np.reshape(penul_output[i], (1,256)) #벡터로 transpose 안되는 경우 있음
    u = np.reshape(mean[label[i]], (1,256))
    Mdist[label[i]].append(distance.mahalanobis(v,u,np.linalg.inv(covar))**2)

"""

# Distance distribution of each class PLT.show
label = []
penul_output = []

label, penul_output = sess.run([tf.argmax(pred, 1),h_fc1], feed_dict={X: mnist.test.images})

plt.rcParams["figure.figsize"] = (10,5)
 
for i in range(10):
    data = np.sort(Mdist[i])
    bins = np.arange(0,2000, 10)  # 도수분포구간 설정
    plt.hist(data, bins, normed=True)
    plt.title("Class: %d" % i)
    plt.xlabel('Distance', fontsize=15)
    plt.ylabel('Num of data', fontsize=15)
    plt.show(block=True)

 
# t-SNE about Test samples    
print("Making t-SNE about test images...")

colors = ['#476A2A', '#7851B8', '#BD3430', '#4A2D4E', '#875525',
          '#A83683', '#4E655E', '#853541', '#3A3120', '#535D8E']
 
tsne = TSNE(random_state=0)
digit = tsne.fit_transform(penul_output)
 
for i in range(len(penul_output)):
    plt.text(digit[i,0], digit[i,1], str(label[i]),
             color = colors[label[i]],
             fontdict={'weight' : 'bold', 'size':9})    
    
plt.xlim(digit[:,0].min(), digit[:,0].max())
plt.ylim(digit[:,1].min(), digit[:,1].max())
plt.xlabel('a')
plt.ylabel('b')
plt.show()
 

tf.reset_default_graph() # 그래프 중복 문제로 restore이 두번째부터는 안되는 현상을 방지 
                        # 내 생각으론 리셋하지 않으면 텐서플로우 안에서 그래프가 중복되어 변수의 개수 증가
                        