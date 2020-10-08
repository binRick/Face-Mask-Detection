#!/usr/bin/env bash
set -e
source source.sh
#args="--confidence 90"
args=""
#cmd="./detect_mask_image.py $args $@"
cmd="./detect_mask_image.py $args $@ 2>/dev/null"
eval $cmd
