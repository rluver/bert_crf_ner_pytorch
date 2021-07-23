# -*- coding: utf-8 -*-
"""
Created on Fri Jul 23 15:22:47 2021

@author: MJH
"""
import pandas as pd
import re
import glob
import json
from tqdm import tqdm



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




def get_input_data(dataset, tokenizer):
    
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
        
        for key in name_entity.keys():
            no_space_key = key.replace(' ', '')
            for find in re.finditer(no_space_key, ''.join(sentence)):                
                index = find.span()
                if ( index[1] - index[0] ) == 1:
                    character_dataframe.loc[index[0], 'tag'] = 'B-' + name_entity[key]
                else:
                    character_dataframe.loc[index[0], 'tag'] = 'B-' + name_entity[key]
                    character_dataframe.loc[( index[0] + 1 ) : (index[1] - 1 ), 'tag'] = 'I-' + name_entity[key]
            
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
        
        input_dataframe = input_dataframe[input_dataframe.text.map(len) <= 98]
        input_dataframe = input_dataframe[input_dataframe.text.apply(lambda x: max([len(i) for i in x]))  <= 18]
        input_dataframe.reset_index(drop = True, inplace = True)
        
    return input_dataframe