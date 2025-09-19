### Config

model_path = 'microsoft_deberta-v3-large'
set_batch_size = 16

#

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

app = FastAPI()

import torch
import json

torch.classes.__path__ = []
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TextClassificationPipeline

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = AutoModelForSequenceClassification.from_pretrained(
    model_path,
    num_labels = 2
    ).to(device)

try:
    tokenizer = AutoTokenizer.from_pretrained(model_path, clean_up_tokenization_spaces = True)
except:
    config = json.load(open(model_path + '/config.json'))
    tokenizer = AutoTokenizer.from_pretrained(config['_name_or_path'], clean_up_tokenization_spaces = True)
    tokenizer.save_pretrained(model_path)

pipe = TextClassificationPipeline(model = model, tokenizer = tokenizer, top_k = None, device = device)

class SentencePair(BaseModel):
    text: str
    text_pair: str

@app.post('/predict')
def predict(pairs: List[SentencePair]):
    template_dict = [{'text': p.text, 'text_pair': p.text_pair} for p in pairs]
    raw_probs = pipe(template_dict, batch_size = set_batch_size)
    return {'raw_probs': raw_probs}