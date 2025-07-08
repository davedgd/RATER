#!/bin/bash

model=`echo $1 | sed 's/\//_/g'`

# source run_full.sh out.csv

date +"### START: %a %b %d %I:%M:%S %p %Z %Y ###"

mamba activate openai
python get_preds.py $model
python eval_preds.py $model
mamba deactivate

date +"### END: %a %b %d %I:%M:%S %p %Z %Y ###"