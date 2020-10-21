#!/usr/bin/env bash
set -e
source concurrent.lib.sh

MAX_QTY=100

get_ids(){
  cat youtube_video_ids.txt|sort -u|shuf|head -n $MAX_QTY
}


cmd="xe -a -j4 -s './fetch_video.sh \"\${1}\"' -- $(get_ids|tr '\n' ' ')"

echo -e $cmd
eval $cmd
exit

gen_concurrent(){
 echo -e "#!/usr/bin/env bash\nsource $(pwd)/concurrent.lib.sh"
 echo -e "concurrent \\"
 echo -e "  - 'My short task' echo OK \\"
 while read -r id; do 
  echo -e "  - '$id' echo ./fetch_video.sh '$id' \\"
 done < <(get_ids)
 echo -e " --require 'My short task' --before-all"
}

gen_concurrent > /tmp/r
chmod +x /tmp/r
cat /tmp/r
#/tmp/r

