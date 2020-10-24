#!/usr/local/bin/bash
set -e
source source.sh
VALID="1 5 10 30"
ENABLED_FPS="1"
set +e
max_duration=600

__h(){
    >&2 echo -e "yyyyyyyyyyyyyyyyyyyyyyyyyyyyy"
    ./killall.sh
}
CONCURRENT_LIMIT="${1:-3}"
shift
trap __h SIGINT SIGTERM

Y_OFFSET=30

cmd="./runit.sh $@"
for index in $(seq 1 $CONCURRENT_LIMIT); do
    Y="$(($index*$Y_OFFSET))"
    Y="$(($Y-$Y_OFFSET))"
    c="$cmd -Y $Y"
    echo -e "$c"
    eval $c &
done

sleep .5

jobs -p
for job in `jobs -p`
do
echo $job
    wait $job || let "FAIL+=1"
done
