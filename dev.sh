#!/bin/bash
set -e
export PATH=$(pwd)/bin:$PATH

nodemon -w . -e py,sh,yaml,yml -x "killall rtsp-simple-server; sleep 1; ./rtsp-simple-server; sleep 3"
