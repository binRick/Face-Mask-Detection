#!/bin/bash
set -e
export PATH=$(pwd)/bin:$PATH

nodemon -w . -e py,sh,yaml,yml -x "while [[ 1 ]]; do killall rtsp-simple-server; sleep 1; ./rtsp-simple-server; sleep 3; done"
