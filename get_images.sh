cmd="ls video_images/newyork/$1_${2}*.png"
eval $cmd |tr '\n' ','
