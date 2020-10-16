[[ -e ./.v/bin/activate ]] || python3 -m venv .v
source ./.v/bin/activate

pip freeze | grep -q tensorflow ||  \
                                    pip install -r requirements.txt
