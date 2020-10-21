#!/bin/bash
set -e
cleanup(){
  killall ffmpeg
}

#trap cleanup EXIT

./get_videos.sh | while read -r vid; do
  name=$(echo -e "$vid"|cut -d'.' -f1|cut -d'/' -f2)
  cmd="while [[ 1 ]]; do ./stream_video.sh '$name' '$vid'; sleep 5; done"
  eval $cmd 2>&1 &
done

sleep 10

for job in `jobs -p`
do
echo $job
    wait $job || let "FAIL+=1"
done

echo $FAIL

if [ "$FAIL" == "0" ];
then
echo "YAY!"
else
echo "FAIL! ($FAIL)"
fi


