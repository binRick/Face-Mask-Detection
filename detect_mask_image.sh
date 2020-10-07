#!/usr/bin/env bash
set -e
source source.sh

cmd="./detect_mask_image.py $@ 2>/dev/null"
cmd="./detect_mask_image.py $@"
eval $cmd
