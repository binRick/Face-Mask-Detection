source source.sh

ffmpeg -re -stream_loop -1 -i $2 -f rtsp -rtsp_transport tcp rtsp://127.0.0.1:8554/$1
