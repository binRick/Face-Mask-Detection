qty="${1:-9}"

cmd="ls videos/*.mkv|egrep -v '.temp.'|tail -n$qty"
eval $cmd
