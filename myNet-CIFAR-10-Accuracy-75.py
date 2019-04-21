import numpy as np
import tensorflow as tf
import os
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.layers import flatten
import matplotlib.pylab as plt
from tensorflow.keras.datasets.cifar10 import load_data


# Set random seed for permutation
np.random.seed(0)
# Set random seed for tf.Variable initialization
tf.set_random_seed(1234)


# Directory(named 'model') for storing trained model.
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'model')
if os.path.exists(MODEL_DIR) is False:
    os.mkdir(MODEL_DIR)

#extract next batch random CIFAR-10 image
def next_batch(num, data, labels):

  idx = np.arange(0 , len(data))
  np.random.shuffle(idx)
  idx = idx[:num]
  data_shuffle = [data[ i] for i in idx]
  labels_shuffle = [labels[ i] for i in idx]

  return np.asarray(data_shuffle), np.asarray(labels_shuffle)


# LeNet-5 from https://github.com/sujaybabruwad/LeNet-in-Tensorflow/blob/master/LeNet-Lab.ipynb
def Mynet(x):
    # Hyperparameters
    mu = 0
    sigma = 0.08
    layer_depth = {'con1': 64, 'con2': 128, 'con3': 256, 'con4': 512, 'L1':128, 'L2':256, 'L3':512, 'L4':1024}

    # Make penultimate logits as global variable
    global fullc2
    
    # size (32, 32, 3) -> (16, 16, 64)
    conv1_w = tf.Variable(tf.truncated_normal(shape=[3, 3, 3, layer_depth.get('con1')], mean=mu, stddev=sigma), name='conv1_w')    
    conv1 = tf.nn.conv2d(x, conv1_w, strides=[1, 1, 1, 1], padding='SAME')  # 'VALID' - without padding
    conv1 = tf.nn.relu(conv1)
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv1_bn = tf.layers.batch_normalization(pool1)
    #conv1 = tf.nn.dropout(conv1, keep_prob)
    
    # size (16, 16, 64) -> (8,8, 128)
    conv2_w = tf.Variable(tf.truncated_normal(shape=[3, 3, layer_depth.get('con1'), layer_depth.get('con2')], mean=mu, stddev=sigma), name='conv2_w')
    conv2 = tf.nn.conv2d(conv1_bn, conv2_w, strides=[1, 1, 1, 1], padding='SAME')
    conv2 = tf.nn.relu(conv2)
    pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv2_bn = tf.layers.batch_normalization(pool2)
    #conv2 = tf.nn.dropout(conv2, keep_prob)
    
 
    
    # size (8,8,128 ) -> (4,4,256)
    conv3_w = tf.Variable(tf.truncated_normal(shape=[5,5,layer_depth.get('con2'),layer_depth.get('con3')], mean=mu, stddev=sigma), name = 'conv3_w')
    conv3 = tf.nn.conv2d(conv2_bn,conv3_w,strides=[1,1,1,1],padding='SAME')
    conv3 = tf.nn.relu(conv3)
    pool3 = tf.nn.max_pool(conv3, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    conv3_bn = tf.layers.batch_normalization(pool3)
    #conv3 = tf.nn.dropout(conv3, keep_prob)
   
    # size (4,4,256) -> (2,2,512)
    conv4_w = tf.Variable(tf.truncated_normal(shape=[5,5,layer_depth.get('con3'),layer_depth.get('con4')], mean=mu, stddev=sigma), name='conv4_w')
    conv4 = tf.nn.conv2d(conv3_bn,conv4_w,strides=[1,1,1,1],padding='SAME')
    conv4 = tf.nn.relu(conv4)
    pool4 = tf.nn.max_pool(conv4, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    conv4_bn = tf.layers.batch_normalization(pool4)
    #conv4 = tf.nn.dropout(conv4, keep_prob)
    
    #conv5 = tf.nn.dropout(conv5, keep_prob)
    
    
    # Flatten -> 2*2*512
    fullc1 = flatten(conv4_bn)
    
    
    # fullc1: Fully-connected ~ (2*2*512) -> (128)
    fullc1 = tf.contrib.layers.fully_connected(inputs=fullc1, num_outputs=128, activation_fn=tf.nn.relu)
    fullc1 = tf.nn.dropout(fullc1, keep_prob)
    fullc1 = tf.layers.batch_normalization(fullc1)
    
    
    # fullc2: Fully-connected ~ (128) -> (256)
    fullc2 = tf.contrib.layers.fully_connected(inputs=fullc1, num_outputs=256, activation_fn=tf.nn.relu)
    fullc2 = tf.nn.dropout(fullc2, keep_prob)
    fullc2 = tf.layers.batch_normalization(fullc2)
    
    
    # fullc3: Fully-connected ~ (256) -> (512)
    fullc3 = tf.contrib.layers.fully_connected(inputs=fullc2, num_outputs=512, activation_fn=tf.nn.relu)
    fullc3 = tf.nn.dropout(fullc3, keep_prob)
    fullc3 = tf.layers.batch_normalization(fullc3)
    
    # logits : (1028) -> (10)
    y = tf.contrib.layers.fully_connected(inputs=fullc3, num_outputs=10, activation_fn=None)
   
    return y


def loss(y, t):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=t)
    loss_function = tf.reduce_mean(cross_entropy)
    return loss_function


def train_step(loss):
    optimizer = tf.train.AdamOptimizer(0.001) #이전의 optimizer 와의 차이를 알아야할듯
    train_step = optimizer.minimize(loss)
    return train_step


def accuracy(y, t):
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(t, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return accuracy

def normalize(inputX):
    Min = np.min(inputX)
    Max = np.max(inputX)
    
    X = (inputX - Min)/(Max - Min)
    
    return X


if __name__ == '__main__':
    # Get data==========================================================================================================
    (X_train, Y_train), (X_test, Y_test) = load_data()
    
    X_train = normalize(X_train)
    X_test = normalize(X_test)
    
    X_valid = X_train[45000:]
    Y_valid = Y_train[45000:]
    X_train = X_train[0:45000]
    Y_train = Y_train[0:45000]
    # get permuted N train data and epoch,batch
    n = len(X_train)-5000
    N = 30000
    epochs = 20
    batch_size = 100
    
    # Set model ========================================================================================================
    x = tf.placeholder(tf.float32, shape=[None, 32, 32, 3]) #RGB 채널 3
    t = tf.placeholder(tf.int32, shape=[None,10])
    keep_prob = tf.placeholder(tf.float32)
    
    y_train_one_hot = tf.squeeze(tf.one_hot(Y_train, 10),axis=1)
    y_test_one_hot = tf.squeeze(tf.one_hot(Y_test, 10),axis=1)
    y_valid_one_hot = tf.squeeze(tf.one_hot(Y_valid, 10), axis = 1)
        
    y = Mynet(x)
    y_pred = tf.nn.softmax(y)
    
    loss = loss(y, t)
    train_step = train_step(loss)
    accuracy = accuracy(y_pred, t)

    # Train and evaluate model =========================================================================================
    #'''
    # 1.Store(Train) ---------------------------------------------------------------------------------------------------
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    sess = tf.Session()
    sess.run(init)
    n_batches = N // batch_size
    for epoch in range(epochs):
        #X_, Y_ = next_batch(N, X_train, y_train_one_hot.eval(session=sess))
        #print(X_.shape)
        #print(Y_.shape)
        for i in range(n_batches):
            X_, Y_ = next_batch(batch_size, X_train, y_train_one_hot.eval(session=sess)) 
            
            sess.run(train_step, feed_dict={x: X_, t: Y_, keep_prob:0.7})           
        
         
        val_loss = loss.eval(session=sess, feed_dict={x:X_valid, t: y_valid_one_hot.eval(session=sess), keep_prob:1.0})
        val_acc = accuracy.eval(session=sess, feed_dict={x:X_valid, t: y_valid_one_hot.eval(session=sess), keep_prob:1.0})
        
        
        #val_loss, val_acc = sess.run([loss,accuracy], feed_dict={x: X_valid, t:y_valid_one_hot.eval(session=sess), keep_prob:1.0})
        print('epoch:', epoch+1, ' loss:', val_loss, ' accuracy:', val_acc)
    # Evaluate accuracy of validation and test datasets
    #val_acc_v = accuracy.eval(session=sess, feed_dict={x: X_validation, t: Y_validation})
    val_acc_t = accuracy.eval(session=sess, feed_dict={x: X_test, t: y_test_one_hot.eval(session=sess), keep_prob:1.0})
    print()
    #print('validation accuracy: {:.4f}'.format(val_acc_v))
    print('test accuracy:       {:.4f}'.format(val_acc_t), end='\n')
    # Store model
    model_path = saver.save(sess, MODEL_DIR + './model.ckpt')
    print('Model saved to:', model_path)
    print()
    # '''
    '''
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
    '''