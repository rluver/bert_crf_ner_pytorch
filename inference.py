# -*- coding: utf-8 -*-
"""
Created on Fri Jul 23 15:38:31 2021

@author: MJH
"""

from tokenization import *
from model import bert_crf_ner
from auxiliary import load_data

import torch
from tensorflow.keras.preprocessing import pad_sequences



def get_name_entity(text, max_len = 256):
    
    text = tokenizer.wordpiece_tokenizer.tokenize(text)
    
    # truncation
    if len(text) > (max_len - 2):
        text = text[:(max_len - 2)]
    text.insert(0, '[CLS]')
    text+= ['[SEP]']
    
    input_ids = tokenizer.convert_tokens_to_ids(text)
    token_type_ids = [[0] * max_len]
    attention_mask = pad_sequences([[1] * len(input_ids)], maxlen = max_len, padding = 'post', value = 0)
    
    input_ids = pad_sequences([input_ids], maxlen = max_len, padding = 'post', dtype = 'int32')
    
    name_entity_token = ner_model(input_ids = input_ids, token_type_ids = token_type_ids, attention_mask = attention_mask)    
    
    return name_entity_token



if  __name__ == '__main__':
    
    tokenizer = BertTokenizer('vocab.korean.rawtext.list')
    
    ner_model = bert_crf_ner.load_from_checkpoint(r'checkpoints\best-checkpoint.ckpt')
    ner_model.freeze()
    
    print('input text: ')
    get_name_entity(input())