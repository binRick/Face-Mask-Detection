qty="${1:-8}"

cmd="ls videos/*.m*|tail -n$qty"
eval $cmd
