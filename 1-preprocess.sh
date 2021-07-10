#!/bin/bash

#cd Neural-Topic-Modeling
#pip install transformers==2.8.0

source activate scholar

python preprocess/download_20ng.py

cd preprocess

python preprocess_data.py \
    ../data/20ng/20ng_all/train.jsonlist \
    processed \
    --vocab-size 5000 \
    --test ../data/20ng/20ng_all/test.jsonlist

mv processed ../data/20ng/processed
cp ../data/20ng/20ng_all/train.jsonlist ../data/20ng/
cp ../data/20ng/20ng_all/test.jsonlist ../data/20ng/

python create_dev_split.py