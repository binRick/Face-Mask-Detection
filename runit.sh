#!/bin/bash
set -e
source source.sh
VALID="1 5 10 30"
ENABLED_FPS="5 10 1"
set +e
max_duration=600

__h(){
    >&2 echo -e "xxxxxxxxxxxxxxxxxxxxxxxxx"
}

trap __h SIGINT SIGTERM

while :; do
 ./get_videos.sh|while read -r video_file; do
  for fps in $ENABLED_FPS; do
   cmd="timeout $max_duration ./detect_mask_video.py -F '$video_file' --fps '$fps' $@"
   echo $cmd
   eval $cmd
   ec=$?
  done 
 done
done
