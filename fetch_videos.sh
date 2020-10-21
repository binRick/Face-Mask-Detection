#!/usr/bin/env bash
set -e
source concurrent.lib.sh
YTARGS="--write-info-json --no-mtime --write-thumbnail  -q"
MAX_QTY=2
get_ids(){
  cat youtube_video_ids.txt|sort -u|shuf|head -n $MAX_QTY
}
#|xargs -n 3 -I % echo -e "% youtube-dl $YTARGS --merge-output-format mkv --max-filesize 500m --limit-rate 2M -o videos/% https://www.youtube.com/watch?v=%"
#}



#concurrent \
#    - 'My long task'   sleep 1 \
#    - 'My medium task' sleep 1  \
#    - 'My short task'  sleep 1  \
#    --require 'My short task' --before-all

youtube_dl(){
  echo -e "$@" >> /tmp/.n
}

gen_concurrent(){
 echo -e "#!/usr/bin/env bash\nsource $(pwd)/concurrent.lib.sh"
 echo -e "youtube_dl(){ youtube-dl $YTARGS --merge-output-format mkv --max-filesize 500m --limit-rate 2M https://www.youtube.com/watch?v=\$1; }"
 echo -e "concurrent \\"
 echo -e "  - 'My short task' echo OK \\"
 while read -r id; do 
  echo -e "  - '$id' youtube_dl '$id' \\"
#  echo -e "  --and-then \\"
 done < <(get_ids)
 echo -e " --require 'My short task' --before-all"
}

gen_concurrent > /tmp/r
chmod +x /tmp/r
/tmp/r

exit

sleep 10

for pid in $(jobs -p); do 
    [[ -d /proc/$pid ]] || continue
    echo $pid
    wait $pid || echo -e "pid $pid failed"
done

