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


# Set random seed for permutation
np.random.seed(0)
# Set random seed for tf.Variable initialization
tf.set_random_seed(1234)

def normalize(inputX):
    Min = np.min(inputX)
    Max = np.max(inputX)
    
    X = (inputX - Min)/(Max - Min)
    
    return X

# CIFAR-10 data load and preprocessing
def data_load():   

    (X_train, Y_train), (X_test, Y_test) = load_data()
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
        
    X_train = normalize(X_train)
    X_test = normalize(X_test)
    
    X_valid = X_train[40000:]
    Y_valid = Y_train[40000:]
    
    X_train = X_train[0:40000]
    Y_train = Y_train[0:40000]
    
    enc = OneHotEncoder()
     
    enc.fit(Y_train)
    y_train_one_hot = enc.transform(Y_train).toarray()
    enc.fit(Y_test)
    y_test_one_hot = enc.transform(Y_test).toarray()
    enc.fit(Y_valid)
    y_valid_one_hot = enc.transform(Y_valid).toarray()
    
    return X_train, y_train_one_hot, X_test, y_test_one_hot, X_valid, y_valid_one_hot

 
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


# loading the normalized CIFAR10 datas
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
                    epochs=20,
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

#result -> test accuracy : 59~61%



