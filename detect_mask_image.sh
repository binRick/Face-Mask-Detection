#!/usr/bin/env bash
set -e
source source.sh
export PATH="$(pwd)/bin:$PATH"

args=""
cmd="./detect_mask_image.py $args $@ 2>/dev/null"
if [[ "1" == "0" ]]; then
 while read -r line; do
  echo -e "GOT LINE '$line'"
  if [[ "$line" == *"dst_image"* ]]; then
    echo -e "  extracting dst_image...."
    dst_image_cmd="echo -e "$line"|jq '.dst_image' -Mrc"
    dst_image="$(eval $dst_image_cmd)"
    echo -e "  $dst_image_cmd"
    echo -e "  $dst_image"
  fi
 done < <(eval $cmd)
else
 eval $cmd
fi
