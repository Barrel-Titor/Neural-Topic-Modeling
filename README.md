# Neural-Topic-Modeling

 Final project for MLDL 2021 in Polito





## 2021-07-06-Zifan

**Rough steps for reimplementing the demo (in Colab)**

1. Install Huggingface transformers of version 2.8.0. Latest version will cause errors.

> pip install transformers==2.8.0



2. Preprocessing on IMDB.

> cd data/imdb/
>
> python download_imdb.py
>
> python preprocess_data.py train.jsonlist processed --vocab-size 5000 --test test.jsonlist
>
> python create_dev_split.py



3. Finetune the teacher model.

> cd ../..
>
> python bert_reconstruction.py \
>
> ​    --input-dir='./data/imdb/processed-dev' \
>
> ​    --output-dir='./data/imdb/processed-dev/logits'





## 2021-05-08-Zifan

This readme file can work as our message board. Sorry for the inconvenience of keeping in touch.
