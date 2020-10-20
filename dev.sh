#!/bin/bash
set -e
export PATH=$(pwd)/bin:$PATH

nodemon -w . -e yml -x "reap -xv ./rtsp-simple-server; sleep 3"
