#!/usr/bin/env bash
set -e

MAX_VIDEO_SIZE=100m
PROXY="--proxy socks5://127.0.0.1:3001/"
YTARGS="--write-info-json --no-mtime --write-thumbnail $PROXY --exec '$(pwd)/process_video.sh $(pwd)/videos/{}'"


cmd="cd videos && youtube-dl $YTARGS --id --recode-video mkv --format 'best[filesize<50M]+best[height<=480]' --merge-output-format mkv --max-filesize $MAX_VIDEO_SIZE --limit-rate 2M https://www.youtube.com/watch?v=$1"

eval $cmd
