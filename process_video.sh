#!/usr/bin/env bash
set -e

F="$(echo -e "$1")"

FPS=1
OUT_DIR=videos_fps/$FPS

OUT_FILE="$OUT_DIR/$(basename "$F"|sed 's/-//g')"

cmd="ffmpeg -i '$F' -r 1 -y '$OUT_FILE'"

echo -e "$cmd" >> ~/.ffmpeg-cmds.txt
echo OK
exit 0

if [[ ! -f "$OUT_FILE" ]]; then
  echo Recoding video to $OUT_FILE
  echo -e  "$cmd"
  eval $cmd
  echo -e "      OK"
else
  echo out file $OUT_FILE OK
fi
