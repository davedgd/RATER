#!/bin/bash

model=`echo $1 | sed 's/\//_/g'`

# source run_full.sh out.csv

date -u +"### START: %a %b %d %I:%M:%S %p %Z %Y ###"

rm -rf model model-awq outputs

mamba activate unsloth-latest
CUDA_VISIBLE_DEVICES=0 python fine_tuning.py $1 $2
mamba deactivate

mamba activate awq
CUDA_VISIBLE_DEVICES=0 python convert_awq.py
mamba deactivate

mamba activate openai
python get_preds.py $model
python eval_preds.py $model
mamba deactivate

mv model-awq ../models/$model
mv $model.csv ../results
rm -rf model outputs

date -u +"### END: %a %b %d %I:%M:%S %p %Z %Y ###"