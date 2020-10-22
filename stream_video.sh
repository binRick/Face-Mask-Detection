source source.sh

df="./analysis_frames/$1-frame.json"
echo $df
[[ ! -f ./analysis_frames/$1-frame.json ]] && 
  ffmpeg -re -stream_loop -1 -i $2 -f rtsp -rtsp_transport tcp rtsp://127.0.0.1:8554/$1
