#!/bin/bash

chmod +x 1-preprocess.sh 2-finetune.sh 3-logits.sh 4-student.sh

./1-preprocess.sh
./2-finetune.sh
#./3-logits.sh
#./4-student.sh