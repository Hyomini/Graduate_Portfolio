import os
import matplotlib.pyplot as plt
import numpy as np
import math
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.datasets.cifar10 import load_data


np.random.seed(0)

from keras import models, layers
from keras import Input
from keras.models import Model, load_model
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers, initializers, regularizers, metrics
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import BatchNormalization, Conv2D, Activation, Dense, GlobalAveragePooling2D
from keras.layers import MaxPooling2D, ZeroPadding2D, ZeroPadding3D, Add
from keras.optimizers import Adam
from keras.regularizers import l2
from keras import backend as K
from keras.layers.core import Lambda
from keras.callbacks import LearningRateScheduler
from keras.models import model_from_json

n = 5
batch_size = 128
n_epoch = 200

def lr_schedule(epoch):
    
    lr = 0.001
    
    if epoch > 150:
        lr *= 0.01
    elif epoch > 100:
        lr *= 0.1
    
    return lr

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
    
    X_valid = X_train[40000:]
    Y_valid = Y_train[40000:]
    
    X_train = X_train[0:40000]
    Y_train = Y_train[0:40000]
    
    X_train = normalize(X_train)
    X_test = normalize(X_test)
    X_valid = normalize(X_valid) # 데이터셋 나누기 전에 정규화, 후에 정규화 하는 것의 차이
    
    enc = OneHotEncoder()
     
    enc.fit(Y_train)
    y_train_one_hot = enc.transform(Y_train).toarray()
    enc.fit(Y_test)
    y_test_one_hot = enc.transform(Y_test).toarray()
    enc.fit(Y_valid)
    y_valid_one_hot = enc.transform(Y_valid).toarray()
    
    return X_train, y_train_one_hot, X_test, y_test_one_hot, X_valid, y_valid_one_hot

def zeropad(x):
    y = K.zeros_like(x)
    return K.concatenate([x, y], axis=3)

def identity_padding(x):
    # block의 feature map 과 channel 수 바뀔때 패딩으로 shortcut을 조정한다
    # (32x32x16) -> (16x16x32) 으로, (16x16x32) -> (8x8x64) 로
    # out_channel - in_channel 만큼 패딩, feature map 은 maxpool로
    
    x = MaxPooling2D()(x)
    x = Lambda(zeropad)(x)
    return x
    
    

def start(x):
    x = ZeroPadding2D(padding=(1,1))(x)
    x = Conv2D(16, (3,3), strides=(1,1), bias=False,
               kernel_initializer="he_normal",
               kernel_regularizer=l2(0.0001))(x)
    x = BatchNormalization(momentum=0.9)(x)
    x = Activation('relu')(x)
    
    return x

def layers_2n(x):
    shortcut = x

    for i in range(n):

        x = Conv2D(16, (3,3), strides=(1,1), padding='same',bias=False)(x)
        x = BatchNormalization(momentum=0.9)(x)
        x = Activation('relu')(x)
            
        x = Conv2D(16, (3,3), strides=(1,1), padding='same',bias=False)(x)
        x = BatchNormalization(momentum=0.9)(x)
    
        x = Add()([x,shortcut])
        x = Activation('relu')(x)
    
        shortcut = x
    return x

def layers_4n(x):
    
    shortcut = x
    strides=(2,2)
    is_identity = 1
    paddings = 'valid'
    
    for i in range(n):
        if is_identity == 1:
            x = ZeroPadding2D(padding=(1,1))(x)
            shortcut = identity_padding(shortcut)
            is_identity = 0
            
        x = Conv2D(32, (3,3), strides=strides, padding=paddings ,bias=False)(x)
        x = BatchNormalization(momentum=0.9)(x)
        x = Activation('relu')(x)
        
        x = Conv2D(32, (3,3), strides=(1,1), padding='same',bias=False)(x)
        x = BatchNormalization(momentum=0.9)(x)
        
        x = Add()([x,shortcut])
        x = Activation('relu')(x)
        
        shortcut = x
        
        strides = (1,1)
        paddings = 'same'
        
    return x            

def layers_6n(x):
    
    shortcut = x
    strides=(2,2)
    is_identity = 1
    paddings = 'valid'
    
    for i in range(n):
        if is_identity == 1:
            x = ZeroPadding2D(padding=(1,1))(x)
            shortcut = identity_padding(shortcut)
            is_identity = 0
            
        x = Conv2D(64 ,(3,3), strides=strides, padding=paddings,bias=False)(x)
        x = BatchNormalization(momentum=0.9)(x)
        x = Activation('relu')(x)
        
        x = Conv2D(64, (3,3), strides=(1,1), padding='same',bias=False)(x)
        x = BatchNormalization(momentum=0.9)(x)
          
        x = Add()([x,shortcut])
        x = Activation('relu')(x)
        
        shortcut = x
        
        strides = (1,1)
        paddings = 'same'
       
    return x


input_shape = Input(shape=(32,32,3), dtype='float32', name='input')

# loading the normalized CIFAR10 datas
X_train, y_train_one_hot, X_test, y_test_one_hot, X_valid, y_valid_one_hot = data_load()

""" 
# when training and saving
# data augmentation -> only horizontal ( training 데이터 좌우반전해서도 학습)
datagen = ImageDataGenerator( featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)

datagen.fit(X_train)

x = start(input_shape)
x = layers_2n(x)
x = layers_4n(x)
x = layers_6n(x)

x = GlobalAveragePooling2D()(x)
output_shape = Dense(10, activation='softmax')(x)

resnet50 = Model(input_shape,output_shape)
resnet50.summary()

resnet50.compile(loss='categorical_crossentropy',optimizer = 'Adam',
              metrics=['accuracy'])

# model saving
model_json = resnet50.to_json()
with open("model.json", "w") as json_file :
    json_file.write(model_json)

# training
print("Training Model...")

history = resnet50.fit_generator(datagen.flow(X_train, y_train_one_hot, batch_size=batch_size),
                                 steps_per_epoch=300,
                    epochs=n_epoch,
                    validation_data = (X_valid,y_valid_one_hot),
                    verbose=2,
                    callbacks=[ LearningRateScheduler(schedule=lr_schedule)]) # verbose -> 학습 진행상황을 어떻게 보여줄까?

resnet50.save_weights("resnet50.h5")
print("Saved model to disk")

print("Evaluating Model...")

(loss, accuracy) = resnet50.evaluate(
        X_test, y_test_one_hot, batch_size=128, verbose=1)

print("accuracy : {:.2f}%".format(accuracy*100))
"""
# augmentation 함수,  imagedatagenerator 에 대한 전처리 과정이 곧 정확도의 열쇠 
# 공부하자

# Just load and evaluate
json_file = open("model.json", "r")
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

loaded_model.load_weights("resnet50.h5")
print("Loaded model from disk")

loaded_model.compile(loss="categorical_crossentropy", optimizer=Adam(lr=lr_schedule(0)), metrics=["accuracy"])

score = loaded_model.evaluate(X_test,y_test_one_hot, verbose=2)

print("%s : %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))
