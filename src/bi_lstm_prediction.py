
# BI-LSTM을 이용한 CS데이터 키워드 도출 모델을 이용한 예측 

# 패키지 불러오기
import logging
import pandas as pd
import numpy as np
import copy
from pathlib import Path
from datetime import datetime
import warnings


## 영어 자연어처리
import re

## 데이터
from collections import Counter
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

## 모델링
from tensorflow.keras.layers import Bidirectional, LSTM, Embedding, Dense
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import OneHotEncoder
import joblib
from tensorflow import random
import random as rn
seed_num = 42
np.random.seed(seed_num)
rn.seed(seed_num)
random.set_seed(seed_num)
from keras import backend as K

warnings.filterwarnings(action='ignore')

logger = logging.getLogger('inter_m') 

dt_now = datetime.now()
date_only = str(dt_now.date())
hour_only = str(dt_now.hour)
min_only = str(dt_now.minute)

class BiLstmModel_Prediction:
    def __init__(self):
        pass

    # 함수 시작 =========================================================== 

    # 독립변수 가공
    def regular_expression(self, dataset):

        # 독립변수(X인자) 가공 : 서비스 내역 1 ~ 5번 병합하여 통합 서비스 내역 구축
        cols = ['서비스내역1', '서비스내역2', '서비스내역3', '서비스내역4','서비스내역5']
        dataset['통합서비스내역'] = dataset[cols].fillna('').apply(lambda row: ' '.join(row.values.astype(str)),axis=1) 

        # 정규식 표현 및 불용어 제거
        temp_dataset = copy.deepcopy(dataset)
        temp_dataset['통합서비스내역_clean'] = np.nan

        for i in range(len(temp_dataset)):
            # 정규 표현식 처리 : re.sub()
            re_sub = temp_dataset.iloc[i,:]['통합서비스내역']
            temp_dataset.iloc[i,-1] = re.sub('[^a-zA-Z가-힣]',' ',re_sub) # 영어 소문자,대문자,한글 # 숫자 제외 

            # capital word transformer
            capital_word = temp_dataset.iloc[i,:]['통합서비스내역_clean']
            temp_dataset.iloc[i,-1] = capital_word.upper()

            # 불용어 제거 
            stop_word = temp_dataset.iloc[i,:]['통합서비스내역_clean']
            temp_dataset.iloc[i,-1] = stop_word.replace('있','').replace('하','').replace('것','').replace('들','').replace('그','').replace('및','').replace('되','').replace('보','').replace('않','').replace('나','').replace('됨','').replace('외','')             


            temp_dataset.iloc[i,-1] = str(temp_dataset.iloc[i,:]['통합서비스내역_clean'])   
        
        return temp_dataset

    # 학습 모델 불러오기
    def model_sel(self):
        model_list = []
        for mod in Path('./model').glob("*.h5"):
            model_list.append(mod)
        #print(model_list[-1])   
        model_list.sort(reverse=True)
        return model_list[-1]   


    # 함수 끝 ===========================================================   


    def run(self):
        start = datetime.now()  

        # 데이터 불러오기
        original_dataset = pd.read_csv('./data/unseen_CS_data.csv') 

        # 독립변수 가공
        dataset = original_dataset
        original_dataset = self.regular_expression(dataset)
        message = '=' * 25+'<1. Independent value set end>'+'=' * 25
        logger.info(message)        

        # 결측치 처리
        remove_set = original_dataset[original_dataset['통합서비스내역_clean'].isnull()].index
        original_dataset = original_dataset.drop(remove_set)
        message = '=' * 25+'<2. Missing data preprocessing end>'+'=' * 25
        logger.info(message)    

        # 독립변수 구성
        X_train = original_dataset['통합서비스내역_clean']
        X_train 

        # Tokenizer
        tokenizer = joblib.load('./weight/tokenizer.joblib')
        X_train_encoded = tokenizer.texts_to_sequences(X_train)
        word_to_index = tokenizer.word_index
        vocab_size = len(word_to_index) + 1
        message = '=' * 25+'<3. Tokenizer end>'+'=' * 25
        logger.info(message)    

        # padding
        max_len = max(len(sample) for sample in X_train_encoded) # X_trsin Row 기준 인코딩 값중 가장 길이가 긴 값 
        X_train_padded = pad_sequences(X_train_encoded, maxlen = max_len) #, maxlen = 47
        message = '=' * 25+'<4. Padding set end>'+'=' * 25
        logger.info(message)    

        # 인코딩 정보 불러오기
        enc = joblib.load('./weight/one_hot_encoding.joblib')   

        # 모델 정보 불러오기
        model_path = self.model_sel()
        modell = load_model(str(model_path))
        message = '=' * 25+'<5. Encoding & Model loading end>'+'=' * 25
        logger.info(message)    

        # 예측하기
        ## X_train 예측 결과
        pred = modell.predict(X_train_padded)
        ## 예측결과 역변화
        reverse_enc = enc.inverse_transform(pred)
        ## 예측 결과값 병합
        pred_check = pd.DataFrame(reverse_enc,columns = ['pred_result'])
        original_dataset_add_pred = pd.concat([original_dataset,pred_check],axis=1)
        message = '=' * 25+'<6. Prediction end>'+'=' * 25
        logger.info(message)    

        # 예측값 저장
        pred_data_save_path = './data/PRED_DATA_BiLSTM.CSV'
        original_dataset_add_pred.to_csv(pred_data_save_path,index=False,encoding='utf-8-sig')
        message = '=' * 25+'<7. Prediction dataset save end>'+'=' * 25
        logger.info(message)    

        print('U R good Enough')    

        print('RUNNING TIME => ',datetime.now() - start)
 