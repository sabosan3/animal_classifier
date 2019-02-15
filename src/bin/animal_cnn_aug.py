#! /usr/bin/python3
# -*- coding:utf-8 -*-

import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.utils import np_utils
from keras.callbacks import EarlyStopping, LearningRateScheduler, ReduceLROnPlateau
import cv2
import numpy as np
import yaml
from progressbar import ProgressBar
import os, time, sys, glob

BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(os.path.dirname(BASE_PATH), "data")

EPOCH = 5000
INIT_LEARN_RATE = 0.001
BATCH_SIZE=32

def main():

  conf_path = os.path.join(BASE_PATH, "conf/application.yaml")
  with open(conf_path, "rb") as f:
    conf = yaml.load(f)

  data_conf = conf["data"]
  classes = data_conf["animal"]
  num_classes = len(classes)

  animal_dir = os.path.join(DATA_PATH, "animal")
  X_train, X_test, Y_train, Y_test = np.load(os.path.join(animal_dir, "animal_aug.npy"))

  X_train = X_train.astype('float') / 256
  X_test = X_test.astype('float') / 256
  Y_train = np_utils.to_categorical(Y_train, num_classes)
  Y_test = np_utils.to_categorical(Y_test, num_classes)

  model = model_train(X_train, Y_train)
  model_eval(model, X_test, Y_test)

def model_train(X, Y):

  model = Sequential()
  model.add(Conv2D(32, (3, 3), padding='same', input_shape=X.shape[1:]))
  model.add(Activation('relu'))
  model.add(Conv2D(32, (3, 3)))
  model.add(Activation('relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Dropout(0.25))

  model.add(Conv2D(64, (3, 3), padding='same'))
  model.add(Activation('relu'))
  model.add(Conv2D(32, (3, 3)))
  model.add(Activation('relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Dropout(0.25))

  model.add(Flatten())
  model.add(Dense(512))
  model.add(Activation('relu'))
  model.add(Dropout(0.5))
  model.add(Dense(3))
  model.add(Activation('softmax'))

  # opt = keras.optimizers.rmsprop(lr=LEARN_RATE, decay=1e-6)
  opt = keras.optimizers.adam(lr=INIT_LEARN_RATE, decay=1e-6)

  early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=50, verbose=1, mode='auto')
  # lr_decay = LearningRateScheduler(step_decay, verbose=1)
  lr_decay = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                patience=5, verbos=1, epsilon=0.1, min_lr=0.00001)

  model.compile(loss='categorical_crossentropy',
                  optimizer=opt, metrics=['accuracy'])

  model.fit(X, Y, batch_size=BATCH_SIZE, epochs=EPOCH, validation_split=0.25, callbacks=[early_stopping, lr_decay])
  
  animal_dir = os.path.join(DATA_PATH, "animal")
  model.save(os.path.join(animal_dir, "animal_aug_adam.h5"))

  return model

def model_eval(model, X, Y):

  scores = model.evaluate(X, Y, verbose=1)
  print("Test Loss: ", scores[0])
  print("Test Accuracy: ", scores[1])

def step_decay(epoch):

  lr = INIT_LEARN_RATE
  if epoch >= round(EPOCH * (1/3)): lr = lr / 10
  if epoch >= round(EPOCH * (2/3)): lr = lr / 10
  return lr 

if __name__=="__main__":
  main()
