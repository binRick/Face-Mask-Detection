#!/usr/bin/env bash
set -e
source source.sh
export PATH="$(pwd)/bin:$PATH"

FRAME_RATE="1"
VIDEO_NAME="$1"
VIDEO_URL="$2"
VIDEO_FILE="videos/$VIDEO_NAME"
VIDEO_IMAGES_DIR="video_images/$VIDEO_NAME"

get_video_images(){
  ls $VIDEO_IMAGES_DIR/${VIDEO_NAME}_*.png
}
get_video_images_qty(){
  get_video_images|wc -l
}

get_video_duration_seconds(){
    cmd="ffprobe -v error -select_streams v:0 -show_entries stream=duration -of default=noprint_wrappers=1:nokey=1 $1"
    eval $cmd | cut -d'.' -f1
}

should_decode_video(){
  VIDEO_DURATION_SECONDS="$1"
  VIDEO_IMAGES_QTY="$2"
  if [[ "$VIDEO_IMAGES_QTY" -gt 0 ]]; then
    false
  else
    true
  fi
}

args=""
cmd="youtube-dl -o $VIDEO_FILE '$VIDEO_URL'"


[[ ! -f $VIDEO_FILE && -f "$VIDEO_FILE.webm" ]] && VIDEO_FILE="$VIDEO_FILE.webm"
[[ ! -f $VIDEO_FILE && -f "$VIDEO_FILE.mp4" ]] && VIDEO_FILE="$VIDEO_FILE.mp4"
[[ ! -f $VIDEO_FILE && -f "$VIDEO_FILE.mkv" ]] && VIDEO_FILE="$VIDEO_FILE.mkv"

[[ -f "$VIDEO_FILE" ]] || eval $cmd

[[ ! -f $VIDEO_FILE && -f "$VIDEO_FILE.webm" ]] && VIDEO_FILE="$VIDEO_FILE.webm"
[[ ! -f $VIDEO_FILE && -f "$VIDEO_FILE.mp4" ]] && VIDEO_FILE="$VIDEO_FILE.mp4"
[[ ! -f $VIDEO_FILE && -f "$VIDEO_FILE.mkv" ]] && VIDEO_FILE="$VIDEO_FILE.mkv"

MSG="Video duration is $(get_video_duration_seconds $VIDEO_FILE) seconds and we have $(get_video_images_qty) images."
echo -e $MSG

[[ -d $VIDEO_IMAGES_DIR ]] || mkdir -p $VIDEO_IMAGES_DIR
cmd="ffmpeg -i $VIDEO_FILE -r $FRAME_RATE $VIDEO_IMAGES_DIR/${VIDEO_NAME}_%05d.png"

if $(should_decode_video "$(get_video_duration_seconds $VIDEO_FILE)" "$(get_video_images_qty)"); then
  echo Should decode video
  [[ -d $VIDEO_IMAGES_DIR ]] && rm -f $VIDEO_IMAGES_DIR/${VIDEO_NAME}_*.png
  eval $cmd
else
  echo Should not decode video
fi





echo "Extracted $(get_video_images | wc -l) images"

cmd="./detect_mask_image.sh -I $(./get_images.sh $VIDEO_NAME)"
eval $cmd
