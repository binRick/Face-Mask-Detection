[[ -e ./.v/bin/activate ]] || python3 -m venv .v
source ./.v/bin/activate

command -v ffmpeg >/dev/null || { 
  sudo dnf -y install epel-release dnf-utils; \
  sudo yum-config-manager --set-enabled PowerTools; \
  sudo yum-config-manager --add-repo=https://negativo17.org/repos/epel-multimedia.repo; \
  sudo dnf -y install ffmpeg;
}

pip freeze | grep -q tensorflow ||  \
                                    pip install -r requirements.txt
