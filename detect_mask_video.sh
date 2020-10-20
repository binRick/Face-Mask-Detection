#!/usr/bin/env bash
set -e
source source.sh
export PATH="$(pwd)/bin:$PATH"

args=""
e="$(mktemp)"
cmd="./detect_mask_video.py $args $@"
eval $cmd
ec=$?
[[ "$ec" != "0" ]] && cat "$e"
[[ -f "$e" ]] && unlink "$e"
