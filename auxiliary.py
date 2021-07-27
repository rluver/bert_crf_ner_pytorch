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


IDS_TO_TAGS = dict(
    {
        0: '[PAD]',
        1: '[UNK]',
        2: '[CLS]',
        3: '[SEP]',
        4: '[MASK]',
        5: 'B-OGG_RELIGION',
        6: 'B-QT_TEMPERATURE',
        7: 'B-PT_FRUIT',
        8: 'B-AFA_VIDEO',
        9: 'B-OGG_ART',
        10: 'B-AM_OTHERS',
        11: 'B-MT_ROCK',
        12: 'B-TM_SPORTS',
        13: 'B-CV_CURRENCY',
        14: 'B-LC_SPACE',
        15: 'B-AF_WEAPON',
        16: 'B-AFA_DOCUMENT',
        17: 'B-LCG_OCEAN',
        18: 'B-AFW_OTHER_PRODUCTS',
        19: 'B-QT_MAN_COUNT',
        20: 'B-TMI_SW',
        21: 'B-TI_MINUTE',
        22: 'B-TMI_MODEL',
        23: 'B-QT_CHANNEL',
        24: 'B-CV_FOOD',
        25: 'B-TR_SCIENCE',
        26: 'B-TMI_SITE',
        27: 'B-PT_GRASS',
        28: 'B-PS_NAME',
        29: 'B-TM_SHAPE',
        30: 'B-OGG_POLITICS',
        31: 'B-EV_FESTIVAL',
        32: 'B-DT_OTHERS',
        33: 'B-TM_DIRECTION',
        34: 'B-OGG_FOOD',
        35: 'B-CV_FOOD_STYLE',
        36: 'B-DT_SEASON',
        37: 'B-CV_TAX',
        38: 'B-FD_MEDICINE',
        39: 'B-AFA_PERFORMANCE',
        40: 'B-OGG_MEDIA',
        41: 'B-MT_CHEMICAL',
        42: 'B-CV_CLOTHING',
        43: 'B-CV_SPORTS_POSITION',
        44: 'B-CV_ART',
        45: 'B-FD_SOCIAL_SCIENCE',
        46: 'B-TR_MEDICINE',
        47: 'B-QT_SPEED',
        48: 'B-QT_OTHERS',
        49: 'B-CV_BUILDING_TYPE',
        50: 'B-TM_CELL_TISSUE_ORGAN',
        51: 'B-PT_TYPE',
        52: 'B-TI_OTHERS',
        53: 'B-QT_PHONE',
        54: 'B-LCP_PROVINCE',
        55: 'B-CV_LANGUAGE',
        56: 'B-LC_OTHERS',
        57: 'B-AF_TRANSPORT',
        58: 'B-CV_POLICY',
        59: 'B-FD_OTHERS',
        60: 'B-MT_METAL',
        61: 'B-QT_PRICE',
        62: 'B-AF_BUILDING',
        63: 'B-DT_MONTH',
        64: 'B-TM_COLOR',
        65: 'B-EV_SPORTS',
        66: 'B-AM_PART',
        67: 'B-QT_LENGTH',
        68: 'B-MT_ELEMENT',
        69: 'B-TMI_HW',
        70: 'B-QT_PERCENTAGE',
        71: 'B-QT_VOLUME',
        72: 'B-CV_LAW',
        73: 'B-LCG_MOUNTAIN',
        74: 'B-TR_SOCIAL_SCIENCE',
        75: 'B-AM_TYPE',
        76: 'B-FD_ART',
        77: 'B-DT_YEAR',
        78: 'B-OGG_LIBRARY',
        79: 'B-PT_TREE',
        80: 'B-TR_OTHERS',
        81: 'B-OGG_OTHERS',
        82: 'B-OGG_MILITARY',
        83: 'B-FD_SCIENCE',
        84: 'B-OGG_LAW',
        85: 'B-AM_MAMMALIA',
        86: 'B-TR_HUMANITIES',
        87: 'B-CV_OCCUPATION',
        88: 'B-OGG_HOTEL',
        89: 'B-QT_COUNT',
        90: 'B-AM_INSECT',
        91: 'B-PS_PET',
        92: 'B-PT_PART',
        93: 'B-QT_WEIGHT',
        94: 'B-PS_CHARACTER',
        95: 'B-TI_HOUR',
        96: 'B-TR_ART',
        97: 'B-CV_SPORTS',
        98: 'B-QT_ORDER',
        99: 'B-EV_OTHERS',
        100: 'B-TMM_DRUG',
        101: 'B-AF_MUSICAL_INSTRUMENT',
        102: 'B-CV_RELATION',
        103: 'B-AF_ROAD',
        104: 'B-CV_POSITION',
        105: 'B-DT_DYNASTY',
        106: 'B-OGG_SCIENCE',
        107: 'B-CV_CULTURE',
        108: 'B-QT_SPORTS',
        109: 'B-DT_WEEK',
        110: 'B-TI_SECOND',
        111: 'B-PT_OTHERS',
        112: 'B-CV_DRINK',
        113: 'B-CV_PRIZE',
        114: 'B-CV_FUNDS',
        115: 'B-TMI_EMAIL',
        116: 'B-OGG_ECONOMY',
        117: 'B-LCP_COUNTY',
        118: 'B-CV_TRIBE',
        119: 'B-QT_AGE',
        120: 'B-AFA_ART_CRAFT',
        121: 'B-TM_CLIMATE',
        122: 'B-LCP_CAPITALCITY',
        123: 'B-LCG_ISLAND',
        124: 'B-AFW_SERVICE_PRODUCTS',
        125: 'B-QT_ALBUM',
        126: 'B-AFA_MUSIC',
        127: 'B-PT_FLOWER',
        128: 'B-AM_BIRD',
        129: 'B-OGG_EDUCATION',
        130: 'B-LCG_CONTINENT',
        131: 'B-AM_AMPHIBIA',
        132: 'B-DT_DURATION',
        133: 'B-EV_ACTIVITY',
        134: 'B-AF_CULTURAL_ASSET',
        135: 'B-LCP_CITY',
        136: 'B-OGG_MEDICINE',
        137: 'B-TI_DURATION',
        138: 'B-LCP_COUNTRY',
        139: 'B-LCG_RIVER',
        140: 'B-CV_SPORTS_INST',
        141: 'B-AM_REPTILIA',
        142: 'B-OGG_SPORTS',
        143: 'B-TMI_SERVICE',
        144: 'B-QT_ADDRESS',
        145: 'B-LCG_BAY',
        146: 'B-TMI_PROJECT',
        147: 'B-QT_SIZE',
        148: 'B-DT_DAY',
        149: 'B-TMM_DISEASE',
        150: 'B-DT_GEOAGE',
        151: 'B-FD_HUMANITIES',
        152: 'B-AM_FISH',
        153: 'B-TMIG_GENRE',
        154: 'B-EV_WAR_REVOLUTION',
        155: 'I-OGG_RELIGION',
        156: 'I-QT_TEMPERATURE',
        157: 'I-PT_FRUIT',
        158: 'I-AFA_VIDEO',
        159: 'I-OGG_ART',
        160: 'I-AM_OTHERS',
        161: 'I-MT_ROCK',
        162: 'I-TM_SPORTS',
        163: 'I-CV_CURRENCY',
        164: 'I-LC_SPACE',
        165: 'I-AF_WEAPON',
        166: 'I-AFA_DOCUMENT',
        167: 'I-LCG_OCEAN',
        168: 'I-AFW_OTHER_PRODUCTS',
        169: 'I-QT_MAN_COUNT',
        170: 'I-TMI_SW',
        171: 'I-TI_MINUTE',
        172: 'I-TMI_MODEL',
        173: 'I-QT_CHANNEL',
        174: 'I-CV_FOOD',
        175: 'I-TR_SCIENCE',
        176: 'I-TMI_SITE',
        177: 'I-PT_GRASS',
        178: 'I-PS_NAME',
        179: 'I-TM_SHAPE',
        180: 'I-OGG_POLITICS',
        181: 'I-EV_FESTIVAL',
        182: 'I-DT_OTHERS',
        183: 'I-TM_DIRECTION',
        184: 'I-OGG_FOOD',
        185: 'I-CV_FOOD_STYLE',
        186: 'I-DT_SEASON',
        187: 'I-CV_TAX',
        188: 'I-FD_MEDICINE',
        189: 'I-AFA_PERFORMANCE',
        190: 'I-OGG_MEDIA',
        191: 'I-MT_CHEMICAL',
        192: 'I-CV_CLOTHING',
        193: 'I-CV_SPORTS_POSITION',
        194: 'I-CV_ART',
        195: 'I-FD_SOCIAL_SCIENCE',
        196: 'I-TR_MEDICINE',
        197: 'I-QT_SPEED',
        198: 'I-QT_OTHERS',
        199: 'I-CV_BUILDING_TYPE',
        200: 'I-TM_CELL_TISSUE_ORGAN',
        201: 'I-PT_TYPE',
        202: 'I-TI_OTHERS',
        203: 'I-QT_PHONE',
        204: 'I-LCP_PROVINCE',
        205: 'I-CV_LANGUAGE',
        206: 'I-LC_OTHERS',
        207: 'I-AF_TRANSPORT',
        208: 'I-CV_POLICY',
        209: 'I-FD_OTHERS',
        210: 'I-MT_METAL',
        211: 'I-QT_PRICE',
        212: 'I-AF_BUILDING',
        213: 'I-DT_MONTH',
        214: 'I-TM_COLOR',
        215: 'I-EV_SPORTS',
        216: 'I-AM_PART',
        217: 'I-QT_LENGTH',
        218: 'I-MT_ELEMENT',
        219: 'I-TMI_HW',
        220: 'I-QT_PERCENTAGE',
        221: 'I-QT_VOLUME',
        222: 'I-CV_LAW',
        223: 'I-LCG_MOUNTAIN',
        224: 'I-TR_SOCIAL_SCIENCE',
        225: 'I-AM_TYPE',
        226: 'I-FD_ART',
        227: 'I-DT_YEAR',
        228: 'I-OGG_LIBRARY',
        229: 'I-PT_TREE',
        230: 'I-TR_OTHERS',
        231: 'I-OGG_OTHERS',
        232: 'I-OGG_MILITARY',
        233: 'I-FD_SCIENCE',
        234: 'I-OGG_LAW',
        235: 'I-AM_MAMMALIA',
        236: 'I-TR_HUMANITIES',
        237: 'I-CV_OCCUPATION',
        238: 'I-OGG_HOTEL',
        239: 'I-QT_COUNT',
        240: 'I-AM_INSECT',
        241: 'I-PS_PET',
        242: 'I-PT_PART',
        243: 'I-QT_WEIGHT',
        244: 'I-PS_CHARACTER',
        245: 'I-TI_HOUR',
        246: 'I-TR_ART',
        247: 'I-CV_SPORTS',
        248: 'I-QT_ORDER',
        249: 'I-EV_OTHERS',
        250: 'I-TMM_DRUG',
        251: 'I-AF_MUSICAL_INSTRUMENT',
        252: 'I-CV_RELATION',
        253: 'I-AF_ROAD',
        254: 'I-CV_POSITION',
        255: 'I-DT_DYNASTY',
        256: 'I-OGG_SCIENCE',
        257: 'I-CV_CULTURE',
        258: 'I-QT_SPORTS',
        259: 'I-DT_WEEK',
        260: 'I-TI_SECOND',
        261: 'I-PT_OTHERS',
        262: 'I-CV_DRINK',
        263: 'I-CV_PRIZE',
        264: 'I-CV_FUNDS',
        265: 'I-TMI_EMAIL',
        266: 'I-OGG_ECONOMY',
        267: 'I-LCP_COUNTY',
        268: 'I-CV_TRIBE',
        269: 'I-QT_AGE',
        270: 'I-AFA_ART_CRAFT',
        271: 'I-TM_CLIMATE',
        272: 'I-LCP_CAPITALCITY',
        273: 'I-LCG_ISLAND',
        274: 'I-AFW_SERVICE_PRODUCTS',
        275: 'I-QT_ALBUM',
        276: 'I-AFA_MUSIC',
        277: 'I-PT_FLOWER',
        278: 'I-AM_BIRD',
        279: 'I-OGG_EDUCATION',
        280: 'I-LCG_CONTINENT',
        281: 'I-AM_AMPHIBIA',
        282: 'I-DT_DURATION',
        283: 'I-EV_ACTIVITY',
        284: 'I-AF_CULTURAL_ASSET',
        285: 'I-LCP_CITY',
        286: 'I-OGG_MEDICINE',
        287: 'I-TI_DURATION',
        288: 'I-LCP_COUNTRY',
        289: 'I-LCG_RIVER',
        290: 'I-CV_SPORTS_INST',
        291: 'I-AM_REPTILIA',
        292: 'I-OGG_SPORTS',
        293: 'I-TMI_SERVICE',
        294: 'I-QT_ADDRESS',
        295: 'I-LCG_BAY',
        296: 'I-TMI_PROJECT',
        297: 'I-QT_SIZE',
        298: 'I-DT_DAY',
        299: 'I-TMM_DISEASE',
        300: 'I-DT_GEOAGE',
        301: 'I-FD_HUMANITIES',
        302: 'I-AM_FISH',
        303: 'I-TMIG_GENRE',
        304: 'I-EV_WAR_REVOLUTION',
        305: 'O'
    }
)
    


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
        
    input_data = get_bert_input_token(text, tokenizer)
    predicted_ner = ner_model(**input_data)
    
    text_token = input_data['input_ids'][0][input_data['input_ids'][0] > 0][1:-1]
    label_tokens = predicted_ner[0][1:-1].cpu()
    
    label = list([IDS_TO_TAGS[label_token.item()] for label_token in label_tokens])
    token_text = tokenizer.convert_ids_to_tokens(text_token.tolist())
    
    entity_list = list(zip(token_text, label))
    entity_list = list(map(lambda x: list(x), entity_list))

    for idx, temp_list in enumerate(entity_list):
        if idx >= len(entity_list) - 1:
            break
        
        # remove impossible cases
        if entity_list[idx][1].startswith('O') and entity_list[idx + 1][1].startswith('I'):
            entity_list[idx + 1][1] = 'O'

    
    ner_dataframe = pd.DataFrame(columns = ['word', 'entity'])
    
    last_entity = entity_list[0][0]
    index = 0
    

    for idx, temp_dict in enumerate(entity_list):
        key, value = temp_dict
            
        if idx >= 1:
            last_entity = entity_list[( idx - 1 )][-1]
        
        key_len = len(key)
                
        if text[index: (index + 1)] == ' ':
            index += 1
            try:
                end_idx += 1
            except:
                pass
            
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
        
        if last_entity.startswith('B') and value.startswith('B'):
            word = text[start_idx: end_idx]
            ner_dataframe = ner_dataframe.append(pd.DataFrame({'word': word, 'entity': last_entity[2:]}, index = [0]))
            
            start_idx = previous_index
            end_idx = index
            
        if last_entity.startswith('I') and value.startswith('B'):
            word = text[start_idx: end_idx]
            ner_dataframe = ner_dataframe.append(pd.DataFrame({'word': word, 'entity': last_entity[2:]}, index = [0]))
            
            start_idx = previous_index
            end_idx = index
                    
        if last_entity.startswith('O') and value.startswith('B'):
            start_idx = previous_index
            end_idx = index
            
        if last_entity.startswith('B') and value.startswith('I'):
            end_idx += key_len
            
        if last_entity.startswith('I') and value.startswith('I'):
            end_idx += key_len
            
        if last_entity.startswith('I') and value.startswith('O'):
            word = text[start_idx: end_idx]
            ner_dataframe = ner_dataframe.append(pd.DataFrame({'word': word, 'entity': last_entity[2:]}, index = [0]))
    
    
    return ner_dataframe
