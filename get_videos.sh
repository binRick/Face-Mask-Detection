qty="${1:-9}"

cmd="ls videos/*.m*|tail -n$qty"
eval $cmd
