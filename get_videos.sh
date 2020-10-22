qty="${1:-9}"

cmd="ls videos/*.mkv|egrep -v '.temp.'|shuf|tail -n$qty"
eval $cmd
