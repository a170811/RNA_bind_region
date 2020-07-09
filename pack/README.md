python3.6 -m venv .env
source .env/bin/activate
pip3 install -r requirements.txt

py main.py --mode all --tr train.csv --va valid.csv --te test.csv
py main.py --mode test --model model.pretrained.h5 --te test.csv
