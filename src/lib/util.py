#! /usr/bin/python3
# -*- coding:utf-8 -*-

import subprocess

def img_show_on_terminal(img_path):

  cmd = "tiv %s" % (img_path)
  subprocess.call(cmd.split())
