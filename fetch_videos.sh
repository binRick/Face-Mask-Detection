#!/usr/bin/env bash
set -e
source concurrent.lib.sh

MAX_QTY=2
CONCURRENT_QTY=1

get_ids(){
  cat youtube_video_ids.txt|sort -u|shuf|head -n $MAX_QTY
}


cmd="xe -a -j$CONCURRENT_QTY -s './fetch_video.sh \"\${1}\"' -- $(get_ids|tr '\n' ' ')"
echo -e $cmd
while :; do
 eval $cmd
 sleep 5
done
