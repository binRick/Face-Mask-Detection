#!/usr/bin/env bash
set -e
source source.sh

MAX_VIDEO_SIZE=100m
YTARGS="--write-info-json --no-mtime --write-thumbnail $PROXY --exec '$(pwd)/process_video.sh \"$(pwd)/videos/{}\"'"


cmd="cd videos && youtube-dl $YTARGS --id --recode-video mkv --format 'best[filesize<100M]+best[height<=480]' --merge-output-format mkv --max-filesize $MAX_VIDEO_SIZE --limit-rate 2M https://www.youtube.com/watch?v=$1"

eval $cmd
