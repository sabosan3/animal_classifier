#! /usr/bin/python3
# -*- coding:utf-8 -*-

from flickrapi import FlickrAPI
from urllib.request import urlretrieve
from pprint import pprint
import yaml
from progressbar import ProgressBar
import os, time, sys

BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(os.path.dirname(BASE_PATH), "data")

WAIT_TIME = 1
PER_PAGE = 400

def download():

  args = sys.argv
  if len(args) < 2 or "-h" in args:
    print("Usage: python3 download.py <Flickr search keyword>")
    sys.exit(0)

  animal_name = args[1]

  conf_path = os.path.join(BASE_PATH, "conf/application.yaml")
  with open(conf_path, "rb") as f:
    conf = yaml.load(f)

  data_conf = conf["data"]
  api_conf = conf["api"]
  
  save_dir = os.path.join(DATA_PATH, "animal/%s" % (animal_name))

  key = api_conf["key"]
  secret = api_conf["secret"]

  flickr = FlickrAPI(key, secret, format='parsed-json')
  response = flickr.photos.search(
    text = animal_name,
    per_page = PER_PAGE,
    media = 'photos',
    sort = 'relevance',
    safe_search = 1,
    extras = 'url_q, licence'
  )

  photos = response['photos']
  # pprint(photos)

  p = ProgressBar(1, PER_PAGE)
  for i, photo in enumerate(photos['photo']):
    url_q = photo['url_q']
    img_name = os.path.join(save_dir, "%s.jpg" % photo['id'])
    
    if os.path.exists(img_name):
      continue
    
    urlretrieve(url_q, img_name)
    
    p.update(i+1)
    time.sleep(WAIT_TIME)

if __name__=="__main__":
  download()
