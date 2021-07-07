# Neural-Topic-Modeling

 Final project for MLDL 2021 in Polito





## 2021-07-06-Zifan

**Rough steps for reimplementing the demo (in Colab)**

1. Install Huggingface transformers of version 2.8.0. Latest version will cause errors.

> cd Neural-Topic-Modeling
>
> pip install transformers==2.8.0



2. Preprocessing on IMDb.

> python data/imdb/download_imdb.py
>
> cd data/imdb/
>
> python preprocess_data.py train.jsonlist processed --vocab-size 5000 --test test.jsonlist
>
> python create_dev_split.py



3. Finetune the teacher model.

> cd ../..
>
> python teacher/bert_reconstruction.py \
>
> ​	--input-dir='./data/imdb/processed-dev' \
>
> ​	--output-dir='./data/imdb/processed-dev/logits'
>
> ​	--do-train \
>
> ​	--evaluate-during-training \
>
> ​	--logging-steps 200 \
>
> ​	--save-steps 1000 \
>
> ​	--num-train-epochs 6 \
>
> ​	--seed 42 \
>
> ​	--num-workers 4 \
>
> ​	--batch-size 20 \
>
> ​	--gradient-accumulation-steps 8 





## 2021-05-08-Zifan

This readme file can work as our message board. Sorry for the inconvenience of keeping in touch.
