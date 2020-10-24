#!/usr/local/bin/bash
set -e
source source.sh
export PATH=/usr/local/Cellar/node/13.11.0/bin:/usr/local/Cellar/node/14.8.0/bin:$PATH
cmd1="./runits.sh $@"


cmd="command multiview"

#eval exec $cmd


#exit



for x in 0 30 60 90; do
 cmd="$cmd [ $cmd1 -X $x ]"
done
cmd="$cmd -x 99960000"

echo $cmd
eval $cmd
exit



sleep .5

jobs -p
for job in `jobs -p`
do
echo $job
    wait $job || let "FAIL+=1"
done
