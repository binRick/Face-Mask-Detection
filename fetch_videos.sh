#!/usr/bin/env bash
set -e
source concurrent.lib.sh

MAX_QTY=55
CONCURRENT_QTY=1

_get_ids(){
  IDS="$(cat youtube_video_ids.txt|sort -u|shuf|head -n $MAX_QTY)"
  echo -e "$IDS"
}

get_ids(){
 _get_ids|while read -r id; do
  if [[ ! -f videos/$id.mkv ]]; then
    echo -e "$id"
  else
    >&2 echo -e "$id OK"
  fi
 done
}

if [[ "$(get_ids)" == "" ]]; then
  echo no ids
fi

cmd="xe -a -j$CONCURRENT_QTY -s './fetch_video.sh \"\${1}\"' -- $(get_ids|tr '\n' ' ')"
echo -e $cmd
while :; do
 eval $cmd
 sleep 5
done
