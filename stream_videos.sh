#!/bin/bash
set -e
cleanup(){
  killall ffmpeg || true
}

#trap cleanup EXIT
#cleanup
bg_pid_file=$(mktemp)
./get_videos.sh | while read -r vid; do
  name=$(echo -e "$vid"|cut -d'.' -f1|cut -d'/' -f2)
  cmd="while [[ 1 ]]; do ./stream_video.sh '$name' '$vid' 2>&1; sleep 5; done"
  eval $cmd 2>&1 &
  bg_pid=$!
  echo -e "$bg_pid" >> $bg_pid_file
done

cat $bg_pid_file

sleep 10
jobs -p
for job in `jobs -p`
do
echo waiting for $job
    wait $job || let "FAIL+=1"
done

echo $FAIL

if [ "$FAIL" == "0" ];
then
echo "YAY!"
else
echo "FAIL! ($FAIL)"
fi

[[ -f "$bg_pid_file" ]] && unlink "$bg_pid_file"
