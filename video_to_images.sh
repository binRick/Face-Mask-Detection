[[ ! -d video_images/$NAME ]] && mkdir -p video_images/$NAME
mkdir video_images/$NAME; time ffmpeg -i  videos/$NAME.mp4 -r 1 video_images/$NAME/${NAME}_%05d.png
