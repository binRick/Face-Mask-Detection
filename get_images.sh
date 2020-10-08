cmd="ls video_images/$1/$1_${2}*.png"
eval $cmd |tr '\n' ','
