#!/bin/bash

get_cmds(){
  cat youtube_video_ids.txt|xargs -n 2 -I % echo -e "youtube-dl --merge-output-format mkv --max-filesize 500m --limit-rate 2M -o videos/% https://www.youtube.com/watch?v=%"
}

while read -r cmd; do 
  eval $cmd &
done < <(get_cmds)

sleep 10

for pid in $(jobs -p); do 
    [[ -d /proc/$pid ]] || continue
    echo $pid
    wait $pid || echo -e "pid $pid failed"
done

