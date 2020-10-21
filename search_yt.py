#!/usr/bin/env python3
import os, json, sys, time
from youtubesearchpython import SearchVideos

SEARCH_TERMS = ['mask styles','how to wear mask','face mask','facemask','n95 mask','my mask']
ids_file = 'youtube_video_ids.txt'

def get_ids():
  with open(ids_file, 'r') as f:
    return [str(l) for l in f.read().splitlines() if len(str(l)) > 0]

acquired_ids = []
DELAY = 0.01

for S in SEARCH_TERMS:
    max_results = 500
    on = 0
    step = 20
    while on < max_results:
      offset = on + 1
      print(f'  on={on}, max={max_results}, step={step}, offset={offset}, term={S}, ')
      search = SearchVideos(S, offset = offset, mode = "json", max_results = step)
      on += step
      ids = [r['id'] for r in dict(json.loads(search.result()))['search_result'] if 'id' in r.keys()]
      for i in ids:
        if not i in acquired_ids:
          acquired_ids.append(i)
      cur_ids = get_ids()
      print(f'  {len(acquired_ids)} acquired_ids, ids={len(ids)}, cur_ids qty={len(cur_ids)}, ')
      for i in acquired_ids:
        if not i in get_ids():
          with open(ids_file, 'a') as f:
            f.write(i+'\n')
      time.sleep(DELAY)
