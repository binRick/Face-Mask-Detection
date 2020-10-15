#!/usr/bin/env bash
set -e
source source.sh
export PATH="$(pwd)/bin:$PATH"

FRAME_RATE="1"
VIDEO_NAME="$1"
VIDEO_URL="$2"
VIDEO_FILE="videos/$VIDEO_NAME.mkv"
VIDEO_IMAGES_DIR="video_images/$VIDEO_NAME"

get_video_images(){
  ls $VIDEO_IMAGES_DIR/${VIDEO_NAME}_*.png
}

args=""
cmd="youtube-dl -o $VIDEO_FILE '$VIDEO_URL'"

[[ -f "$VIDEO_FILE" ]] || eval $cmd

[[ -d $VIDEO_IMAGES_DIR ]] && rm -f $VIDEO_IMAGES_DIR/${VIDEO_NAME}_*.png
[[ -d $VIDEO_IMAGES_DIR ]] || mkdir -p $VIDEO_IMAGES_DIR

cmd="ffmpeg -i $VIDEO_FILE -r $FRAME_RATE $VIDEO_IMAGES_DIR/${VIDEO_NAME}_%05d.png"
eval $cmd

echo "Extracted $(get_video_images | wc -l) images"

cmd="./detect_mask_image.sh -I $(./get_images.sh $VIDEO_NAME)"
eval $cmd
