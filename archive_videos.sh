#!/bin/bash
set -e


./get_videos.sh | while read -r vid; do
  name=$(echo -e "$vid"|cut -d'.' -f1|cut -d'/' -f2)
  cmd="time borg list .borg::$name || time borg create .borg::$name videos/$name.* --stats --progress"
  echo -e "$cmd"
  eval $cmd
done
