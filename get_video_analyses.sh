qty="${1:-30}"
px=400

cmd="ls analysis_frames/*-frame-analysis-x${px}.json \
  |egrep -v '.temp.'|shuf|tail -n$qty"
eval $cmd
