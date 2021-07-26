# -*- coding: utf-8 -*-
"""
Created on Fri Jul 23 15:22:47 2021

@author: MJH
"""
import pandas as pd
import torch
import re
import glob
import json
from tensorflow.keras.preprocessing import pad_sequences
from tqdm import tqdm


with open('ids_to_tags.txt', 'r') as f:
    IDS_TO_TAGS = eval(f.read())



def load_data(file_path, sep = '\t'):
    '''    

    Parameters
    ----------
    file_path : str
        dataset directory
    sep : TYPE, optional
        DESCRIPTION. The default is '\t'.

    Returns
    -------
    dictionary : TYPE
        DESCRIPTION.

    this load function is for corpus korean dataset

    '''
    
    file_lists = glob.glob('/'.join([file_path, '*.JSON']))
    
    dictionary = []
    
    for file in tqdm(file_lists):
        with open(file, 'r', encoding = 'utf-8') as f:
            json_file = json.loads(f.read()).get('document')
            
            for _json in json_file:
                _json = list(filter(lambda x: x['ne'] != [], _json.get('sentence')))
                for datum in _json:
                    ne_dictionary = dict()
                    sentence = datum.get('form')
                    dic = ((doc.get('form'), doc.get('label')) for doc in datum.get('ne'))
                    
                    for word, tag in dic:
                        ne_dictionary[word] = tag
                    
                    temp = {'sentence': sentence, 'ne': ne_dictionary}
                    
                    dictionary.append(temp)
                    
    return dictionary




def get_input_data(dataset, tokenizer, MAX_LEN):
    
    input_dataframe = pd.DataFrame(columns = ['text', 'tags'])
    
    for data in tqdm(dataset):
        sentence = data['sentence']
        sentence = sentence.replace('\xad', '­＿')
        name_entity = data['ne']
        
        if name_entity == {}:
            continue
        
        bert_tokenized_sentence = tokenizer.wordpiece_tokenizer.tokenize(sentence)
        sentence = bert_tokenized_sentence
        
        character_dataframe = pd.DataFrame([j for i in sentence for j in i], columns = ['text'])
        
        try:
            for key in name_entity.keys():
                no_space_key = key.replace(' ', '')
                for find in re.finditer(no_space_key, ''.join(sentence)):                
                    index = find.span()
                    if ( index[1] - index[0] ) == 1:
                        character_dataframe.loc[index[0], 'tag'] = 'B-' + name_entity[key]
                    else:
                        character_dataframe.loc[index[0], 'tag'] = 'B-' + name_entity[key]
                        character_dataframe.loc[( index[0] + 1 ) : (index[1] - 1 ), 'tag'] = 'I-' + name_entity[key]
        
        except:
            continue
        
        character_dataframe.fillna('O', inplace = True)
        
        start = 0
        bert_tag_list = []
        for token in bert_tokenized_sentence:
            bert_tag_list.append((token, start, len(token)))
            start += len(token)
        
        try:
            temp_dict = [{'name': row[0], 'tag': character_dataframe.iloc[row[1]].tag} for row in bert_tag_list]
        except:
            continue
        
        bert_tag = list(map(lambda x: x['tag'], temp_dict))
        
        input_dataframe = input_dataframe.append(pd.DataFrame([[sentence, bert_tag]], columns = ['text', 'tags']))
        
        input_dataframe = input_dataframe[input_dataframe.text.map(len) <= ( MAX_LEN - 2 )]
        input_dataframe = input_dataframe[input_dataframe.text.apply(lambda x: max([len(i) for i in x]))  <= 18]
        input_dataframe.reset_index(drop = True, inplace = True)
        
    return input_dataframe


def get_bert_input_token(text, tokenizer, max_len = 256):
    
    text = tokenizer.wordpiece_tokenizer.tokenize(text)
    
    # truncation
    if len(text) > (max_len - 2):
        text = text[:(max_len - 2)]
    text.insert(0, '[CLS]')
    text += ['[SEP]']
    
    input_ids = tokenizer.convert_tokens_to_ids(text)
    attention_mask = pad_sequences([[1] * len(input_ids)], maxlen = max_len, padding = 'post')
    token_type_ids = [[0] * max_len]
    
    input_ids = pad_sequences([input_ids], maxlen = max_len, padding = 'post', dtype = 'int32')
    
    return dict(
        input_ids = torch.tensor(input_ids).long(),
        token_type_ids = torch.tensor(token_type_ids).long(),
        attention_mask = torch.tensor(attention_mask).long()
    )


def get_entity(text, ner_model, tokenizer):
        
    input_data = get_bert_input_token(text)
    predicted_ner = ner_model(**input_data)
    
    text_token = input_data['input_ids'][0][input_data > 0][1:-1]
    label_tokens = predicted_ner[0][1:-1].cpu()
    
    label = list([IDS_TO_TAGS[label_token.item()] for label_token in label_tokens])
    token_text = tokenizer.convert_ids_to_tokens(text_token.tolist())
    
    entity_list = list(zip(token_text, label))
    entity_list = list(map(lambda x: list(x), entity_list))

    for idx, temp_list in enumerate(entity_list):
        if idx >= len(entity_list) - 1:
            break
        
        if entity_list[idx][1].startswith('O') and entity_list[idx + 1][1].startswith('I'):
            entity_list[idx + 1][1] = 'O'

    
    ner_dataframe = pd.DataFrame(columns = ['word', 'entity'])
    
    last_entity = entity_list[0][0]
    index = 0
    start_idx = 0
    end_idx = 0
    

    for idx, temp_dict in enumerate(entity_list):
        key, value = temp_dict
            
        if idx >= 1:
            last_entity = entity_list[( idx - 1 )][-1]
        
        key_len = len(key)
        
        
        if text[index: (index + 1)] == ' ':
            index += 1
            
            if key == text[index: (index + key_len)]:
                previous_index = index
                index += key_len
    
        else:
            if key == text[index: (index + key_len)]:
                previous_index = index
                index += key_len        
            else:
                previous_index += key_len
                index += key_len
 
    
        if idx == 0 and value.startswith('B'):
            start_idx = previous_index
            end_idx = index
            print(start_idx, end_idx)
        
        if last_entity.startswith('B') and value.startswith('B'):
            end_idx = index
            word = text[start_idx: end_idx]
            ner_dataframe = ner_dataframe.append(pd.DataFrame({'word': word, 'entity': value[2:]}, index = [0]))
            
            start_idx = previous_index
            end_idx = index
            print(start_idx, end_idx)
                    
        if last_entity.startswith('O') and value.startswith('B'):
            start_idx = previous_index
            end_idx = index
            print(start_idx, end_idx)
            
        if last_entity.startswith('B') and value.startswith('I'):
            end_idx += key_len
            print(start_idx, end_idx)
            
        if last_entity.startswith('I') and value.startswith('I'):
            end_idx += key_len
            print(start_idx, end_idx)    
            
        if last_entity.startswith('I') and value.startswith('O'):
            end_idx += key_len
            print(start_idx, end_idx)
            word = text[start_idx: end_idx]
            ner_dataframe = ner_dataframe.append(pd.DataFrame({'word': word, 'entity': last_entity[2:]}, index = [0]))
    
    
    return ner_dataframe
