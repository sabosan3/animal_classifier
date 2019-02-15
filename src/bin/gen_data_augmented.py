#! /usr/bin/python3
# -*- coding:utf-8 -*-

import cv2
import numpy as np
from sklearn import model_selection
import yaml
from progressbar import ProgressBar
import os, time, sys, glob

BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(os.path.dirname(BASE_PATH), "data")

IMAGE_SIZE = 50
IMAGE_NUM = 200
TEST_NUM = 100

def main():

  conf_path = os.path.join(BASE_PATH, "conf/application.yaml")
  with open(conf_path, "rb") as f:
    conf = yaml.load(f)

  data_conf = conf["data"]
  classes = data_conf["animal"]

  X_train = []
  X_test = []
  Y_train = []
  Y_test = []
  for idx, class_label in enumerate(classes):
    img_dir = os.path.join(DATA_PATH, "animal/%s" % (class_label))
    img_files = glob.glob(os.path.join(img_dir, "*.jpg"))

    p = ProgressBar(1, IMAGE_NUM)
    for i, img_file in enumerate(img_files):
      
      if i >= 200:
        break

      img = cv2.imread(img_file, cv2.IMREAD_COLOR)
      img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))

      if i < TEST_NUM:
        X_test.append(img)
        Y_test.append(idx)
      else:
        # X_train.append(img)
        # Y_train.append(idx)

        for angle in range(-20, 20, 5):
          center = (img.shape[1]*0.5, img.shape[0]*0.5)
          rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
          img_r = cv2.warpAffine(img, rot_mat, (img.shape[1],img.shape[0]), flags=cv2.INTER_LINEAR)
          X_train.append(img_r)
          Y_train.append(idx)

          img_trans = cv2.flip(img_r, 1)
          X_train.append(img_trans)
          Y_train.append(idx)

      p.update(i+1)

  # X = np.array(X)
  # Y = np.array(Y)

  X_train = np.array(X_train)
  X_test = np.array(X_test)
  Y_train = np.array(Y_train)
  Y_test = np.array(Y_test)

  # X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y)
  XY = (X_train, X_test, Y_train, Y_test)

  save_dir = os.path.join(DATA_PATH, "animal")
  np.save(os.path.join(save_dir, "animal_aug.npy"), XY)

if __name__=="__main__":
  main()
