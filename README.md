# Neural-Topic-Modeling

 Final project for MLDL 2021 in Polito





## 2021-07-06-Zifan

**Rough steps for reimplementing the demo (in Colab)**

1. Install Huggingface transformers of version 2.8.0. Latest version will cause errors.

```bash
cd Neural-Topic-Modeling

pip install transformers==2.8.0
```



2. Preprocessing on IMDb.

```bash
python data/imdb/download_imdb.py

cd data/imdb/

python preprocess_data.py train.jsonlist processed --vocab-size 5000 --test test.jsonlist

python create_dev_split.py
```



3. Finetune the teacher model.

```bash
cd ../..

python teacher/bert_reconstruction.py \
	--input-dir='./data/imdb/processed-dev' \
	--output-dir='./data/imdb/processed-dev/logits' \
	--do-train \
	--evaluate-during-training \
	--logging-steps 200 \
	--save-steps 1000 \
	--num-train-epochs 6 \
	--seed 42 \
	--num-workers 4 \
	--batch-size 20 \
	--gradient-accumulation-steps 8 
```



4. Extract logits from the teacher model.

```bash
python teacher/bert_reconstruction.py \
    --output-dir ./data/imdb/processed-dev/logits \
    --seed 42 \
    --num-workers 6 \
    --get-reps \
    --checkpoint-folder-pattern "checkpoint-9000" \
    --save-doc-logits \
    --no-dev
```



## 2021-05-08-Zifan

This readme file can work as our message board. Sorry for the inconvenience of keeping in touch.
