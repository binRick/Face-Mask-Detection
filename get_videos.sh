qty="${1:-2}"

cmd="ls videos/*.mkv|egrep -v '.temp.'|shuf|tail -n$qty"
eval $cmd
