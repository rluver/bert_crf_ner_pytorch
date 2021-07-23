# -*- coding: utf-8 -*-
"""
Created on Fri Jul 23 15:29:36 2021

@author: MJH
"""
import itertools
import torch
import torch.nn as nn
import pytorch_lightning as pl
from sklearn.metrics import f1_score
from transformers import BertConfig, BertForTokenClassification
from transformers.optimization import AdamW, get_linear_schedule_with_warmup




class bert_crf_ner(pl.LightningModule):
    
    def __init__(self, train_samples, batch_size, epochs, num_labels):
        super().__init__()
    
        self.train_samples = train_samples
        self.batch_size = batch_size
        self.gradient_accumulation_steps = 1
        self.epochs = epochs
        self.warm_up_proportion = 0.2
        self.num_train_optimization_steps = int(self.train_samples / self.batch_size / self.gradient_accumulation_steps) * epochs
        self.num_warmup_steps = int(float(self.num_train_optimization_steps) * self.warm_up_proportion)

        config = BertConfig.from_pretrained('model', output_hidden_states = True)
        config.num_labels = num_labels
        self.bert_model = BertForTokenClassification.from_pretrained('model', config = config)
        
        self.optimizer_grouped_parameters = self.get_optimizer_grouped_parameters()
        self.dropout = nn.Dropout(p = 0.5)
        self.linear_layer = nn.Linear(in_features = config.hidden_size, out_features = config.num_labels)
        self.crf = CRF(num_tags = config.num_labels, batch_first = True)
        
        
    def forward(self, input_ids, token_type_ids, attention_mask, labels = None):
        outputs = self.bert_model(
            input_ids = input_ids,
            token_type_ids = token_type_ids,
            attention_mask = attention_mask,
            labels = None
            )
                                
        dropout_layer = self.dropout(outputs.hidden_states[-1])
        linear_layer = self.linear_layer(dropout_layer)
                
        if labels is not None:
            log_likelihood = -self.crf(linear_layer, labels, mask = attention_mask.bool(), reduction = 'token_mean')
            sequence_of_tags = torch.tensor(self.crf.decode(linear_layer, mask = attention_mask.bool()))
            
            return log_likelihood, sequence_of_tags
        
        else:
            sequence_of_tags = torch.tensor(self.crf.decode(linear_layer, mask = attention_mask.bool())).cuda()
            
            return sequence_of_tags

         
    def training_step(self, batch, batch_index):
        input_ids = batch['input_ids']
        token_type_ids = batch['token_type_ids']
        attention_mask = batch['attention_mask']
        tags = batch['label'].long()
                
        log_likelihood, sequence_of_tags = self(
            input_ids = input_ids,
            token_type_ids = token_type_ids,
            attention_mask = attention_mask,
            labels = tags
            )
        
        sequence_of_tags = torch.tensor(list(itertools.chain(*sequence_of_tags)))
        real = tags[attention_mask.bool()].cpu()
        correct_num = torch.sum(sum(torch.eq(real, sequence_of_tags)))
        total_num = attention_mask.bool().sum()
        acc = correct_num / total_num
        f1score = f1_score(real.cpu(), sequence_of_tags, average = 'macro')
        
        self.log('train_acc', acc, prog_bar = True, logger = True)
        self.log('train_f1', f1score, prog_bar = True, logger = True)            
        
        return log_likelihood
        
    
    
    def validation_step(self, batch, batch_index):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        token_type_ids = batch['token_type_ids']
        tags = batch['label'].long()

        
        log_likelihood, sequence_of_tags = self(
            input_ids = input_ids,
            token_type_ids = token_type_ids,
            attention_mask = attention_mask,
            labels = tags
            )
                
        sequence_of_tags = torch.tensor(list(itertools.chain(*sequence_of_tags)))
        real = tags[attention_mask.bool()].cpu()
        correct_num = torch.sum(sum(torch.eq(real, sequence_of_tags)))
        total_num = attention_mask.bool().sum()
        acc = correct_num / total_num
        f1score = f1_score(real.cpu(), sequence_of_tags, average = 'macro')
        
        return {'val_loss': log_likelihood, 'val_acc': acc, 'val_f1': f1score}
        
    
    
    def test_step(self, batch, batch_index):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        token_type_ids = batch['token_type_ids']
        tags = batch['label']

        log_likelihood, sequence_of_tags = self.bert_model(
            input_ids = input_ids,
            token_type_ids = token_type_ids,
            attention_mask = attention_mask,
            labels = tags
            )
                
        sequence_of_tags = torch.tensor(list(itertools.chain(*sequence_of_tags)))
        real = tags[attention_mask.bool()].cpu()
        correct_num = torch.sum(sum(torch.eq(real, sequence_of_tags)))
        total_num = attention_mask.bool().sum()
        # acc = correct_num / total_num
        # f1score = f1_score(real.cpu(), sequence_of_tags, average = 'macro')

        return log_likelihood
    
            
    def get_optimizer_grouped_parameters(self):
                
        no_decay_layer_list = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        
        optimizer_grouped_parameters = []

        layers = list(self.bert_model.named_parameters())
        
        encoder_decay = {
            'params': [param for name, param in layers if
                       not any(no_decay_layer_name in name for no_decay_layer_name in no_decay_layer_list)],
            'weight_decay': 0.01
            }
    
        encoder_nodecay = {
            'params': [param for name, param in layers if
                       any(no_decay_layer_name in name for no_decay_layer_name in no_decay_layer_list)],
            'weight_decay': 0.0}
        
        optimizer_grouped_parameters.append(encoder_decay)
        optimizer_grouped_parameters.append(encoder_nodecay)
            
        return optimizer_grouped_parameters
    
    
    
    def configure_optimizers(self):
        
        optimizer = AdamW(
            self.optimizer_grouped_parameters,            
            correct_bias = False
            )

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps = self.num_warmup_steps,
            num_training_steps = self.num_train_optimization_steps
            )
        
        return [optimizer], [{'scheduler': scheduler, 'interval': 'step'}]