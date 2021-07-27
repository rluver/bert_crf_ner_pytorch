# -*- coding: utf-8 -*-
"""
Created on Fri Jul 23 15:38:31 2021

@author: MJH
"""

from tokenization import *
from model import bert_crf_ner
from auxiliary import EntityExtractor


if __name__ == '__main__':
    
    tokenizer = BertTokenizer('vocab.korean.rawtext.list')
    
    ner_model = bert_crf_ner.load_from_checkpoint(r'checkpoints\best-checkpoint.ckpt')
    ner_model.freeze()
    
    entity_extractor = EntityExtractor(ner_model, tokenizer, 256)
    
    text = ''
    entity_extractor.get_entity(text)
    
        
    
    
    