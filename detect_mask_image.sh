#!/usr/bin/env bash
set -e
cmd="./detect_mask_image.py $@ 2>/dev/null"
eval $cmd
