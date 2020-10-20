youtube-dl -o - "$2" | ffmpeg -re -stream_loop -1 -i - -c copy -f rtsp rtsp://127.0.0.1:8554/$1
