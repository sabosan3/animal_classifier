#! /usr/bin/python3
# -*- coding:utf-8 -*-

from flask import Flask
import os

BASE_PATH = os.path.basename(os.path.basename(os.path.abspath(__file__)))

app = Flask(__name__)

@app.route('/')
def hello_world():

  return 'Hello World!'
