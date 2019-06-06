#!/bin/bash

python3 train.py -c -ns -np
python3 train.py -c -s -np

python3 train.py -h -ns -np
python3 train.py -h -s -np

python3 predict.py -c -ns -np
python3 predict.py -c -s -np

python3 predict.py -h -ns -np
python3 predict.py -h -s -np
