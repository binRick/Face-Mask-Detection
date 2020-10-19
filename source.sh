[[ -e ./.v/bin/activate ]] || python3 -m venv .v
source ./.v/bin/activate

[[ ! -f ./rtsp-simple-server ]] || {wget https://github.com/aler9/rtsp-simple-server/releases/download/v0.10.1/rtsp-simple-server_v0.10.1_darwin_amd64.tar.gz; }

command -v ffmpeg >/dev/null || { 
  sudo dnf -y install epel-release dnf-utils; \
  sudo yum-config-manager --set-enabled PowerTools; \
  sudo yum-config-manager --add-repo=https://negativo17.org/repos/epel-multimedia.repo; \
  dnf clean packages; \
  sudo dnf -y install ffmpeg;
}

pip freeze | grep -q tensorflow ||  \
                                    pip install -r requirements.txt
