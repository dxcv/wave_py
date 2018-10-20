# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19 11:04:13 2018
脉象波形分类训练
@author: haoqi
"""

import matplotlib.pyplot as plt
import numpy as np
import keras
from keras.layers import Input, Flatten, Dense, MaxPooling1D, Dropout
from keras.layers.convolutional import Conv1D
from keras.utils.np_utils import to_categorical
from keras.models import Model
from train_utils import load_dataset

# 导入数据
length_wave = len(wave)
train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes = load_dataset()
x_train = train_set_x_orig.reshape(-1,length_wave,1)
x_test = test_set_x_orig.reshape(-1,length_wave,1)
#x_train = train_set_x_orig
#x_test = test_set_x_orig
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
#x_train /= 255
#x_test /= 255 
y_train = keras.utils.to_categorical(train_set_y_orig.squeeze(), 4)
y_test = keras.utils.to_categorical(test_set_y_orig.squeeze(), 4)

# build the model
x = Input(shape=(700,1))
y = Conv1D(16,kernel_size=7,strides=1,padding='valid',activation='relu',kernel_initializer='uniform')(x)
y = MaxPooling1D(pool_size=2)(y)

y = Conv1D(32,kernel_size=3,strides=1,padding='valid',activation='relu',kernel_initializer='uniform')(y)
y = MaxPooling1D(pool_size=2)(y)

y = Conv1D(64,kernel_size=3,strides=1,padding='valid',activation='relu',kernel_initializer='uniform')(y)
y = Conv1D(64,kernel_size=3,strides=1,padding='valid',activation='relu',kernel_initializer='uniform')(y)
y = MaxPooling1D(pool_size=2)(y)

y = Flatten()(y)
y = Dense(64,activation='relu')(y)
y = Dropout(0.5)(y)
y = Dense(32,activation='relu')(y)
y = Dropout(0.5)(y)
y = Dense(4,activation='softmax')(y)

model = Model(inputs=x, outputs=y, name='model')
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()

h = model.fit(x_train, y_train, epochs=200, batch_size=8, shuffle=True)
acc = h.history['acc']
m_acc = np.argmax(acc)
print("@ Best Training Accuracy: %.2f %% achieved at EP #%d." % (acc[m_acc] * 100, m_acc + 1))
loss, accuracy = model.evaluate(x_test, y_test, batch_size=20)
print('test loss: ',loss, 'test accuracy: ', accuracy)