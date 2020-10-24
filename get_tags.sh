ls videos/*.info.json|xargs -I % echo -e "cat %| jq -Mrc '.tags'"|bash|egrep -v '^\[\]$' \
    |sed 's/"//g'|sed 's/\[//g'|sed 's/\]//g'|tr ',' '\n'
