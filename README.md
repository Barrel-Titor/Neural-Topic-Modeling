# Neural-Topic-Modeling

 Final project for MLDL 2021 in Polito





## 2021-07-09 Zifan

**Rough steps for working on 20ng dataset**

1. Install Huggingface transformers of version 2.8.0. Latest version will cause errors.

```bash
cd Neural-Topic-Modeling

pip install transformers==2.8.0
```



2. Preprocessing on 20ng.

```bash
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
```



3. Finetune the teacher model.

```bash
cd ..

python -m torch.distributed.launch \
	--nproc_per_node 8 \
	teacher/bert_reconstruction.py \
    --input-dir='./data/20ng/processed-dev' \
    --output-dir='./data/20ng/processed-dev/logits' \
    --do-train \
    --evaluate-during-training \
    --logging-steps 200 \
    --save-steps 1000 \
    --num-train-epochs 50 \
    --seed 42 \
    --num-workers 4 \
    --batch-size 20 \
    --gradient-accumulation-steps 8 
```



4. Extract logits from the teacher model.

```bash
python -m torch.distributed.launch \
	--nproc_per_node 8 \
	teacher/bert_reconstruction.py \
    --output-dir ./data/20ng/processed-dev/logits \
    --seed 42 \
    --num-workers 6 \
    --get-reps \
    --checkpoint-folder-pattern "checkpoint-9000" \
    --save-doc-logits \
    --no-dev
```



5. Run the base topic model and perform knowledge distillation.

```bash
python scholar/run_scholar.py \
    ./data/20ng/processed-dev \
    --dev-metric npmi \
    -k 50 \
    --epochs 500 \
    --patience 500 \
    --batch-size 200 \
    --background-embeddings \
    --device 0 \
    --dev-prefix dev \
    -lr 0.002 \
    --alpha 0.5 \
    --eta-bn-anneal-step-const 0.25 \
    --doc-reps-dir ./data/20ng/processed-dev/logits/checkpoint-9000/doc_logits \
    --use-doc-layer \
    --no-bow-reconstruction-loss \
    --doc-reconstruction-weight 0.5 \
    --doc-reconstruction-temp 1.0 \
    --doc-reconstruction-logit-clipping 10.0 \
    -o ./outputs/20ng
```





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
