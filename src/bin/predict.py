
#! /usr/bin/python3
# -*- coding:utf-8 -*-

import keras
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.utils import np_utils
import cv2
import numpy as np
import yaml
from progressbar import ProgressBar
import os, time, sys, glob

BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(os.path.dirname(BASE_PATH), "data")

IMG_SIZE = 50
CHANEL = 3
EPOCH = 100
LEARN_RATE = 0.0001
BATCH_SIZE=32

def build_model():

  model = Sequential()
  model.add(Conv2D(32, (3, 3), padding='same', input_shape=(IMG_SIZE, IMG_SIZE, CHANEL)))
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

  opt = keras.optimizers.rmsprop(lr=LEARN_RATE, decay=1e-6)
  
  model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

  model_dir = os.path.join(DATA_PATH, "animal")
  model = load_model(os.path.join(model_dir, "animal.h5"))

  return model

def main():

  args = sys.argv
  if len(args) < 2 or "-h" in args:
    print("Usage: python3 predict.py <path/to/image_file>")
    sys.exit(0)

  conf_path = os.path.join(BASE_PATH, "conf/application.yaml")
  with open(conf_path, "rb") as f:
    conf = yaml.load(f)

  data_conf = conf["data"]
  classes = data_conf["animal"]  

  img = cv2.imread(args[1], cv2.IMREAD_COLOR)
  img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
  img = img.astype('float') / 256

  X = np.array([img])
  model = build_model()

  result = model.predict([X])[0]
  predicted = result.argmax()
  percentage = int(result[predicted] * 100)
  print("{0} ({1}%)".format(classes[predicted], percentage))

if __name__=="__main__":
  main()
