# CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server --model model-awq --max_model_len 4096 --gpu_memory_utilization .95

# Notes

#'gpt-4o-2024-08-06' -- reported model (no ft)
#'gpt-3.5-turbo-0125' -- reported model (no ft)
#'gpt-4o-mini-2024-07-18' -- reported model (no ft)
#'ft:gpt-4o-2024-08-06:dobolyilab:ht:A3rU60He' -- reported model (auto ft: epochs 3; batch size 1; LR multiplier 2; seed 3407)
#'ft:gpt-3.5-turbo-0125:dobolyilab:ht:A3rafxKq' -- reported model (auto ft: epochs 3; batch size 1; LR multiplier 2; seed 3407)
#'ft:gpt-4o-mini-2024-07-18:dobolyilab:ht:A3rD3mRe' -- reported model (auto ft: epochs 3; batch size 1; LR multiplier 1.8; seed 3407)

# Configure Basic Settings

auto_launch_vllm = True
use_openai = False
chosen_model = 'model-awq'
cuda_devices = '0'
openai_api_key = '...' # enter a valid OpenAI API key
set_max_workers = 30

# Load Packages

import os, subprocess, signal, time
import pandas as pd
from progressbar import progressbar
from openai import OpenAI
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import sys

import janitor
import numpy as np

# Configure Test Data and Model Settings

set_temperature = 0
set_max_tokens = 128
set_seed = 1234
set_extra_body = {
    "guided_choice": [
        "1. Item does an EXTREMELY BAD job of measuring the concept provided above",
        "2. Item does a VERY BAD job of measuring the concept provided above",
        "3. Item does a SOMEWHAT BAD job of measuring the concept provided above",
        "4. Item does an ADEQUATE job of measuring the concept provided above",
        "5. Item does a SOMEWHAT GOOD job of measuring the concept provided above",
        "6. Item does a VERY GOOD job of measuring the concept provided above",
        "7. Item does an EXTREMELY GOOD job of measuring the concept provided above",
        ]}

test = pd.read_excel('../data/processed/train_val_test.xlsx', sheet_name = 'test')[['definition', 'ItemText', 'Target']].clean_names()

thePerson = '''You are an academic expert. Please follow all the instructions very carefully. The questions are unique to survey measurement development and require detailed attention.

Research projects often use survey items to measure concepts. Examples in the management field include work motivation, job satisfaction, and employee stress. When writing survey items, researchers must take great care to ensure that the items do a good job of measuring the concepts of interest (e.g., that an item intended to measure work motivation really seems to capture that concept well). Your purpose is to assess survey items used in the various literatures (e.g., management).'''

thePrompt = '''### Your job is to assess the degree to which each survey item matches the concept statement provided.

You will be given a concept statement below, followed by a survey item. For each item, you will rate the degree to which it matches the provided concept statement.

Not all of the survey items will match the provided concept statement. Therefore, please pay close attention to each individual survey item as you decide whether it matches the provided concept statement.

### You will judge how well a survey item matches a particular statement using this response scale:
1. Item does an EXTREMELY BAD job of measuring the concept provided above
2. Item does a VERY BAD job of measuring the concept provided above
3. Item does a SOMEWHAT BAD job of measuring the concept provided above
4. Item does an ADEQUATE job of measuring the concept provided above
5. Item does a SOMEWHAT GOOD job of measuring the concept provided above
6. Item does a VERY GOOD job of measuring the concept provided above
7. Item does an EXTREMELY GOOD job of measuring the concept provided above

### The concept statement:
{statement}

### The survey item:
{item}

### Answer the question by immediately stating one response from the scale above verbatim. YOUR RESPONSE MUST MATCH THE SCALE EXACTLY WITHOUT ALTERATIONS, INCLUDING THE SCALE RESPONSE NUMBER.'''

#print(thePrompt)

theJustification = 'LASTLY, PROVIDE A JUSTIFICATION FOR YOUR CHOICE.'
thePromptWithJustification = thePrompt + ' ' + theJustification

prompts = []

for index, row in progressbar(test.iterrows(), max_value = len(test)):
    prompt = thePrompt.format(
        statement = row['definition'], 
        item = row['itemtext'])
    prompts += [prompt]

###

if use_openai:
    client = OpenAI(
        api_key = openai_api_key,
        max_retries = 10,
    )
    set_extra_body = {}
else:
    client = OpenAI(
        api_key = 'EMPTY',
        base_url = 'http://localhost:8000/v1',
    )
    set_extra_body = set_extra_body

    if auto_launch_vllm:
        
        command = 'CUDA_VISIBLE_DEVICES=' + cuda_devices + ' mamba run -n vllm python -m vllm.entrypoints.openai.api_server --model ' + chosen_model + ' --max_model_len 4096 --gpu_memory_utilization .95'
        p = subprocess.Popen(command, shell = True, stdout = subprocess.DEVNULL, stderr = subprocess.STDOUT)
        
        while True:
            status, output = subprocess.getstatusoutput('curl http://localhost:8000/v1/models')
            if status == 0:
                break
            else:
                time.sleep(10)
                print('Waiting for vLLM...')

def answer_prompt (person, prompt, model, temperature, max_tokens, seed, extra_body):
    chat_response = client.chat.completions.create(
        model = model, 
        messages = [
            {"role": "system", "content": person},
            {"role": "user", "content": prompt}
        ],
        temperature = temperature,
        max_tokens = max_tokens,
        seed = seed,
        extra_body = extra_body,
        logprobs = True,
        timeout = 60,
    )

    return [chat_response.choices[0].message.content, np.exp(np.mean([token.logprob for token in chat_response.choices[0].logprobs.content]))]

def run_inference (each_prompt):
    res = answer_prompt(
        person = thePerson, 
        prompt = each_prompt,
        model = chosen_model,
        temperature = set_temperature,
        max_tokens = set_max_tokens,
        seed = set_seed,
        extra_body = set_extra_body
        )
    return(res)

with ThreadPoolExecutor(max_workers = set_max_workers) as executor:
    res = list(tqdm(executor.map(run_inference, prompts), total = len(prompts)))

resp = [x[0] for x in res]
prob = [x[1] for x in res]

try:
    pred = [int(resp[0]) for resp in resp]
except:
    print('### NOTE: Issue with integer coding -- manual recode required! ###')
    pred = resp

df = pd.DataFrame({
    'target': test['target'],
    'result': res,
    'pred': pred,
    'prob': prob
    })

df.to_csv(sys.argv[1].replace('/', '_') + '.csv', index = False)

if 'p' in globals():
    os.killpg(os.getpgid(p.pid), signal.SIGTERM)