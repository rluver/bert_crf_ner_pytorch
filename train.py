# -*- coding: utf-8 -*-
"""
Created on Fri Jul 23 15:28:47 2021

@author: MJH
"""
from tokenization import *
from auxiliary import load_data, get_input_data
from model import bert_crf_ner

import pandas as pd
import numpy as np
import os
import parmap
import torch
import itertools
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import Dataset, DataLoader
from tensorflow.keras.preprocessing import pad_sequences
from tqdm import tqdm
tqdm.pandas()




class NERDataset(Dataset):
    
    def __init__(
            self,
            data: pd.DataFrame,
            tokenizer: BertTokenizer,
            text_max_token_length: int = 256
            ):
                
        self.tokenizer = tokenizer
        self.data = data
        self.text_max_token_length = text_max_token_length
        
    
    def __len__(self):
        
        return len(self.data)
    
    
    def _get_bert_input_data(self, text):
                
        # truncation
        input_ids = self.tokenizer.convert_tokens_to_ids(text)
        if len(input_ids) > (self.text_max_token_length - 2):
            input_ids = input_ids[:(self.text_max_token_length - 2)]
        input_ids = list(itertools.chain(*[[2], input_ids, [3]]))
                    
        attention_mask = pad_sequences([[1] * len(input_ids)], maxlen = self.text_max_token_length, padding = 'post')
        segment_ids = [[0] * self.text_max_token_length]
        
        input_ids = pad_sequences([input_ids], maxlen = self.text_max_token_length, padding = 'post', dtype = 'int32')
        
        return dict(
            input_ids = torch.tensor(input_ids), 
            attention_mask = torch.tensor(attention_mask),
            segment_ids = torch.tensor(segment_ids)
            )
    
    
    def __getitem__(self, index: int):
        
        data_row = self.data.iloc[index]    
        encoded_text = self._get_bert_input_data(data_row['text'])
        
        return dict(
            input_ids = encoded_text['input_ids'].flatten().long(),
            token_type_ids = encoded_text['segment_ids'].flatten().long(),
            attention_mask = encoded_text['attention_mask'].flatten().long(),
            label = torch.tensor(data_row.tags)
            )
    
    
    

class NERDataModule(pl.LightningDataModule):
    
    def __init__(            
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        tokenizer: BertTokenizer,
        batch_size: int = 64,
        text_max_token_length: int = 256,
    ):
        
        super().__init__()
        
        self.train_df = train_df
        self.test_df = test_df
        
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.text_max_token_length = text_max_token_length
        
        self.setup()
        
        
    def __len__(self):
        return len(self.train_df)
        


    def setup(self, stage = None):
        self.train_dataset = NERDataset(
            self.train_df,
            self.tokenizer,
            self.text_max_token_length,
            )
        
        self.test_dataset = NERDataset(
            self.test_df,
            self.tokenizer,
            self.text_max_token_length,
            )
    
    
    def train_dataloader(self):        
        return DataLoader(
            self.train_dataset,
            batch_size = self.batch_size,
            shuffle = False
            )

    
    def val_dataloader(self):        
        return DataLoader(
            self.test_dataset,
            batch_size = self.batch_size,
            shuffle = False
            )
    
    
    def test_dataloader(self):        
        return DataLoader(
            self.test_dataset,
            batch_size = self.batch_size,
            shuffle = False
            )
    
    


def main(EPOCHS, BATCH_SIZE, MAX_LEN):
    
    tokenizer = BertTokenizer('vocab.korean.rawtext.list')
    dataset = load_data('corpus_korean')
    
    num_cores = os.cpu_count() - 2
    
    splitted_dataset = np.array_split(dataset, num_cores)    
    temp_input_dataframe = parmap.map(get_input_data, splitted_dataset, tokenizer, MAX_LEN, pm_pbar = True, pm_processes = num_cores)
    input_dataframe = pd.DataFrame()
    for dataframe in temp_input_dataframe:
        input_dataframe = input_datafame.append(dataframe)
    input_dataframe.reset_index(drop = True, inplace = True)
    
    tags = set(itertools.chain(*[set(i.values()) for i in list(map(lambda x: x['ne'], dataset))]))
    tags = list(map(lambda x: 'B-' + x, tags)) + list(map(lambda x: 'I-' + x, tags))
    
    tags_to_ids = {c: ( i + 5 ) for i, c in enumerate(tags)}
    tags_to_ids['O'] = 305
    tags_to_ids['[PAD]'] = tokenizer.vocab.get('[PAD]')
    tags_to_ids['[UNK]'] = tokenizer.vocab.get('[UNK]')
    tags_to_ids['[CLS]'] = tokenizer.vocab.get('[CLS]')
    tags_to_ids['[SEP]'] = tokenizer.vocab.get('[SEP]')
    tags_to_ids['[MASK]'] = tokenizer.vocab.get('[MASK]')
    
    ids_to_tags = {}
    for key, value in tags_to_ids.items():
        ids_to_tags[value] = key
        
    input_dataframe.tags = input_dataframe.tags.progress_apply(lambda x: ['[CLS]'] + x + ['[SEP]'])
    input_dataframe.tags = input_dataframe.tags.progress_apply(lambda x: [tags_to_ids[i] for i in x])
    input_dataframe.tags = input_dataframe.tags.progress_apply(lambda x: pad_sequences([x], max_len = MAX_LEN, padding = 'post', value = tags_to_ids['[PAD]'][0]))
    
    train, test = train_test_split(input_dataframe, test_size = 0.2)

    data_module = NERDataModule(train, test, tokenizer, batch_size = BATCH_SIZE)    

    checkpoint_callback = ModelCheckpoint(
        dirpath = 'checkpoints',
        filename = 'best-checkpoint',
        save_top_k = 1,
        verbose = True,
        monitor = 'val_loss',
        mode = 'min'        
        )
    early_stopping = EarlyStopping(
        monitor = 'val_loss',
        patience = 2,
        mode = 'min'
        )
    
    logger = TensorBoardLogger('lightning_logs', name = 'name_entity_recognition')
        
    trainer = pl.Trainer(
        logger = logger,
        callbacks = [checkpoint_callback, early_stopping],
        max_epochs = EPOCHS,
        progress_bar_refresh_rate = 1,
        gpus = 2,
        accelerator = 'dp'
        )
    
    model = bert_crf_ner(train_samples = train.shape[0], batch_size = BATCH_SIZE, epochs = EPOCHS, num_labels = len(tags_to_ids))

    trainer.fit(model, data_module)
    
    
    
    

if __name__ == '__main__':
            
    EPOCHS = 10
    BATCH_SIZE = 128
    MAX_LEN = 256
    
    main(EPOCHS, BATCH_SIZE, MAX_LEN)