#! /usr/bin/python3
# -*- coding:utf-8 -*-

from flask import Flask, flash, request, redirect, url_for
import os
from werkzeug.utils import secure_filename

from keras.models import Sequential, load_model
import keras
import sys, yaml
import numpy as np
import cv2

BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(os.path.dirname(BASE_PATH), "data")

UPLOAD_FOLDER = os.path.join(DATA_PATH, "uploads")
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

IMAGE_SIZE = 50

with open(os.path.join(BASE_PATH, "conf/application.yaml"), "rb") as f:
  conf = yaml.load(f)
CLASSES = conf["data"]["animal"]
NUM_CLASSES = len(CLASSES)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):

  return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():

  if request.method == 'POST':

    if 'file' not in request.files:
      flash('ファイルがありません')
      return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
      flash('ファイルがありません')
      return redirect(request.url)

    if file and allowed_file(file.filename):
      filename = secure_filename(file.filename)
      filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
      file.save(filepath)
      
      model = load_model(os.path.join(DATA_PATH, "animal/animal.h5"))

      img = cv2.imread(filepath)
      img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
      img = img / 255
      X = np.array([img])
      
      result = model.predict([X])[0]
      predicted = result.argmax()
      percentage = int(result[predicted] * 100)

      return CLASSES[predicted] + str(percentage) + " %"
      
      # return redirect(url_for('uploaded_file', filename=filename))
  
  return '''
  <! doctype html>
  <html><head>
  <meta charset="UTF-8">
  <title>ファイルをアップロードして判定します<></title><head>
  <body>
  <h1>ファイルをアップロードして判定します！</h1>
  <form method = post enctype = multipart/form-data>
  <p><input type=file name=file>
  <input type=submit value=Upload>
  </form>
  </body>
  </html>
  '''

from flask import send_from_directory

@app.route(UPLOAD_FOLDER + "/<filename>")
def uploaded_file(filename):

  return send_from_directory(app.config['UPLOAD_FOLDER'], filename)
