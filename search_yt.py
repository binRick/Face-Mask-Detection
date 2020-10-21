#!/usr/bin/env python3
import os, json, sys, time
from youtubesearchpython import SearchVideos

def get_ids():
  with open('youtube_video_ids.txt','r') as f:
    return f.read().splitlines()

max_results = 2000
on = 0
step = 50
while on < max_results:
  search = SearchVideos("mask styles", offset = on+1, mode = "json", max_results = step)
  for r in dict(json.loads(search.result()))['search_result']:
    mat = (str(r['id']) in get_ids())
    print(r['id'], get_ids(), mat)
    if not str(r['id']) in get_ids():
      with open('youtube_video_ids.txt','a') as f:
        f.write(str(r['id'])+"\n")
  on += step
  print(get_ids())
  
  time.sleep(1)
#print(dict(json.loads(search.result()))['search_result'][0])
