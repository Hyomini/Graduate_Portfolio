from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers import BatchNormalization
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import SGD

from tensorflow.keras.datasets.cifar10 import load_data
from sklearn.preprocessing import OneHotEncoder
from keras.callbacks import ModelCheckpoint
from keras.models import model_from_json

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import scipy.io
import os

# Set random seed for permutation
np.random.seed(0)
# Set random seed for tf.Variable initialization
tf.set_random_seed(1234)

# Directory(named 'model') for storing trained model.
MODEL_DIR = os.path.join(os.path.dirname(__file__))
if os.path.exists(MODEL_DIR) is False:
    os.mkdir(MODEL_DIR)

def normalize(inputX):
    Min = np.min(inputX)
    Max = np.max(inputX)
    
    X = (inputX - Min)/(Max - Min)
    
    return X

# SVHN data load and preprocessing
def data_load():
    
    Train = scipy.io.loadmat(MODEL_DIR + "/train_32x32.mat")
    Test = scipy.io.loadmat(MODEL_DIR + "/test_32x32.mat")
     
    Train_x = Train['X']
    Train_y = Train['y']
     
    # svhn 사진 ( 레이블 y 배열 제외 ) 은 (32, 32, 3, None) 으로 되어있으므로
    # 해왔던 모델을 사용할거면 열 교환이 필요
    Train_x = Train_x.transpose(3,0,1,2)
     
    # train 사진 중 23257 장은 validation set 으로 쓰고, 나머지 50000장 만 training 
    Valid_x = Train_x[50000:,:,:,:]
    Valid_y = Train_y[50000:,:]
     
    Train_x = Train_x[0:50000,:,:,:]
    Train_y = Train_y[0:50000,:]
     
     
    Test_x = Test['X']
    Test_x = Test_x.transpose(3,0,1,2)
    Test_y = Test['y']
     
    Train_x = normalize(Train_x)
    Valid_x = normalize(Valid_x)
    Test_x = normalize(Test_x)
     
    for i in range(50000):
        
        if Train_y[i] == 10:
            Train_y[i] = 0
            
    for i in range(23257):
        
        if Valid_y[i] == 10:
            Valid_y[i] = 0
            
    for i in range(26032):
        
        if Test_y[i] == 10:
            Test_y[i] = 0                
     
    enc = OneHotEncoder()
     
    enc.fit(Train_y)
    y_train_one_hot = enc.transform(Train_y).toarray()
    enc.fit(Test_y)
    y_test_one_hot = enc.transform(Test_y).toarray()
    enc.fit(Valid_y)
    y_valid_one_hot = enc.transform(Valid_y).toarray()
    
    return Train_x, y_train_one_hot, Test_x, y_test_one_hot, Valid_x, y_valid_one_hot

#Making Model
def Lenet(width, height, depth, classes, weightsPath=None):
    
    model = Sequential()
    
    #레이어 형성
    
    # Conv1 + RELU + MaxPool1 ( 32x32x3 -> 16x16x6 )
    model.add(Conv2D(6, (5,5), padding='same', input_shape=(height,width,depth)))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
    model.add(Dropout(0.25))
    
    # Conv2 + RELU + Maxpool2 ( 16x16x6 -> 8x8x16)
    model.add(Conv2D(16, (5,5),padding='same'))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
    model.add(Dropout(0.25))
    
    # FC1 + RELU ( 8x8x16(1024) -> 120)
    model.add(Flatten())
    model.add(Dense(120))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    
    # FC2 + RELU ( 120 -> 84)
    model.add(Dense(84))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    
    # softmax classifier ( 84 -> 10)
    model.add(Dense(10))
    model.add(Activation("softmax"))
    
    if weightsPath is not None:
        model.load_weights(weightsPath)
    
    return model


# SVHN 데이터 로드 및 정규화
X_train, y_train_one_hot, X_test, y_test_one_hot, X_valid, y_valid_one_hot = data_load()


# build and compile model for SAVING
    
print("Building and compiling Lenet Model...")
opt = SGD(lr=0.05)
model = Lenet(width = 32, height = 32, depth = 3, classes = 10)
model.compile(loss="categorical_crossentropy",
              optimizer=opt, metrics=["accuracy"])

model_json = model.to_json()
with open("model.json", "w") as json_file :
    json_file.write(model_json)

print("Training Model...")

history = model.fit(X_train, y_train_one_hot,
                    batch_size = 128,
                    epochs=30,
                    validation_data = (X_valid,y_valid_one_hot),
                    verbose=2 ) # verbose -> 학습 진행상황을 어떻게 보여줄까?

model.save_weights("model.h5")
print("Saved model to disk")

print("Evaluating Model...")

(loss, accuracy) = model.evaluate(
        X_test, y_test_one_hot, batch_size=128, verbose=1)

print("accuracy : {:.2f}%".format(accuracy*100))


""" # just LOADING model and evaluate

json_file = open("model.json", "r")
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

loaded_model.load_weights("model.h5")
print("Loaded model from disk")

loaded_model.compile(loss="categorical_crossentropy", optimizer=SGD(lr=0.1), metrics=["accuracy"])

score = loaded_model.evaluate(X_test,y_test_one_hot, verbose=2)

print("%s : %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))

"""







