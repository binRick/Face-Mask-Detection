[[ -e ./.v/bin/activate ]] || python3 -m venv .v
source ./.v/bin/activate

[[ -d .tmp ]] && rm -rf .tmp

[[ ! -f ./rtsp-simple-server ]] && uname -a | grep -qi linux && { echo Linux; wget https://github.com/aler9/rtsp-simple-server/releases/download/v0.10.1/rtsp-simple-server_v0.10.1_linux_amd64.tar.gz; mkdir .tmp; cd .tmp; tar zxvf ../rtsp-simple-server_v0.10.1_linux_amd64.tar.gz; mv rtsp-simple-server ../; cd ../; rm -rf .tmp;} 
[[ ! -f ./rtsp-simple-server ]] && uname -a | grep -qi darwin && { wget https://github.com/aler9/rtsp-simple-server/releases/download/v0.10.1/rtsp-simple-server_v0.10.1_darwin_amd64.tar.gz; mkdir .tmp; cd .tmp; tar zxvf ../rtsp-simple-server_v0.10.1_darwin_amd64.tar.gz; mv rtsp-simple-server ../; cd ../; rm -rf .tmp; }

command -v ffmpeg >/dev/null || { 
  sudo dnf -y install epel-release dnf-utils; \
  sudo yum-config-manager --set-enabled PowerTools; \
  sudo yum-config-manager --add-repo=https://negativo17.org/repos/epel-multimedia.repo; \
  dnf clean packages; \
  sudo dnf -y install ffmpeg;
}

pip freeze | grep -q tensorflow ||  \
                                    pip install -r requirements.txt

uname -a | grep -qi darwin && alias borg="$(pwd)/bin/borg-macosx"

export BORG_REPO="$(pwd)/.borg"
[[ ! -d .borg ]] && borg init -e repokey


PROXY=""
PROXY="--proxy socks5://127.0.0.1:3001/"

export PROXY
