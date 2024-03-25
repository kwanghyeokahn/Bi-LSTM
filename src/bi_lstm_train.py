
# BI-LSTM을 이용한 CS데이터 키워드 도출 모델 학습 

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

from src.utils.inter_m_config import InterMConfig
seed_num = 42
np.random.seed(seed_num)
rn.seed(seed_num)
random.set_seed(seed_num)
from keras import backend as K

#import os 
#current_path = os.getcwd()

warnings.filterwarnings(action='ignore')

logger = logging.getLogger('inter_m') 

dt_now = datetime.now()
date_only = str(dt_now.date())
hour_only = str(dt_now.hour)
min_only = str(dt_now.minute)

class BiLstmTrain:
    def __init__(self):
        self.original_dataset = None
    # 함수 시작 ===========================================================

    # 데이터 수작업 처리
    def _handmade_transfer(self, target,transfer):
        target = self.original_dataset[self.original_dataset['핵심어_modified2'] == target].index
        for i in range(len(target)):
            self.original_dataset.loc[target[i],'핵심어_modified2'] = transfer

    # 자연어 처리
    def _regular_expression(self):
        
        # 독립변수(X인자) 가공 : 서비스 내역 1 ~ 5번 병합하여 통합 서비스 내역 구축
        cols = ['서비스내역1', '서비스내역2', '서비스내역3', '서비스내역4','서비스내역5']
        self.original_dataset['통합서비스내역'] = self.original_dataset[cols].fillna('').apply(lambda row: ' '.join(row.values.astype(str)),axis=1)

        # 정규식 표현 및 불용어 제거
        # self.temp_dataset = copy.deepcopy(self.original_dataset)
        self.original_dataset['통합서비스내역_clean'] = np.nan
        
        for i in range(len(self.original_dataset)):
            # 정규 표현식 처리 : re.sub()
            re_sub = self.original_dataset.iloc[i,:]['통합서비스내역']
            self.original_dataset.iloc[i,-1] = re.sub('[^a-zA-Z가-힣]',' ',re_sub) # 영어 소문자,대문자,한글 # 숫자 제외
            # capital word transformer
            capital_word = self.original_dataset.iloc[i,:]['통합서비스내역_clean']
            self.original_dataset.iloc[i,-1] = capital_word.upper()
            # 불용어 제거 
            stop_word = self.original_dataset.iloc[i,:]['통합서비스내역_clean']
            self.original_dataset.iloc[i,-1] = stop_word.replace('있','').replace('하','').replace('것','').replace('들','').replace('그','').replace('및','').replace('되','').replace('보','').replace('않','').replace('나','').replace('됨','').replace('외','')             
                                
            self.original_dataset.iloc[i,-1] = str(self.original_dataset.iloc[i,:]['통합서비스내역_clean'])


    def _core_word_data_upsampling(self):
        
        # DATA_SET = copy.deepcopy(self.original_dataset)
        
        temp_dataset = self.original_dataset[['통합서비스내역_clean','핵심어_modified2']]
        x = temp_dataset['통합서비스내역_clean']
        y = temp_dataset['핵심어_modified2']

        x = np.array(x).reshape(len(x), 1)
        y = np.array(y)

        oversample = RandomOverSampler(sampling_strategy='minority')
        x_over, y_over = oversample.fit_resample(x, y)
        
        for i in range(len(Counter(y))):
            oversample = RandomOverSampler(sampling_strategy='minority')
            x_over, y_over = oversample.fit_resample(x_over, y_over)
            
        return x_over, y_over




    # 함수 끝 ===========================================================


    def run(self):
        start = datetime.now()
        train_data_path = InterMConfig().train_data_path
        self.original_dataset = pd.read_csv(train_data_path)
        
        remove_set = self.original_dataset[self.original_dataset['핵심어_modified2'].isnull()].index
        self.original_dataset = self.original_dataset.drop(remove_set)

        logger.info('{0}{1}{2}'.format('=' * 25, '<1. Missing data preprocessing end>', '=' * 25))

        target = 'B’D'
        transfer = "B'D"
        self._handmade_transfer(target,transfer)
        logger.info('{0}{1}{2}'.format('=' * 25, '<2. Uncommon data preprocessing end >', '=' * 25))


        self._regular_expression()


        # 데이터 불균형 처리 : upsampling(minority 한 unique 종속변수 제거의 현실적 어려움으로 모든 종속변수 활용을 위해 최대 unique 종속변수 갯수를 기준으로 단순 upsampling
        x_over, y_over = self._core_word_data_upsampling()
        logger.info('{0}{1}{2}'.format('=' * 25,'<3. Data imbalanced preprocessing end>','=' * 25))

        # 종속변수(Y인자) 원핫인코딩 : get_dummies
        y_over_df = pd.DataFrame(y_over)
        y_over_df.columns= ['core_word']

        enc = OneHotEncoder(sparse=False)
        enc.fit(y_over_df[['core_word']])
        
        enc_name = 'one_hot_encoding'
        en_save_path = InterMConfig().encoding_save_path   
        joblib.dump(enc, en_save_path+'['+date_only+hour_only+min_only+']_'+enc_name+'.joblib')

        en_load_path = InterMConfig().encoding_load_path
        enc = joblib.load(en_load_path)
        y_over_dumm = enc.transform(y_over_df[['core_word']])
        logger.info('{0}{1}{2}'.format('=' * 25,'<4. Dependent value one-hot encoding end>','=' * 25))

        # 데이터 분리(train_test_split)
        x_over = pd.DataFrame(x_over).squeeze()
        X_train, X_test, y_train, y_test = train_test_split(x_over, y_over_dumm, test_size=0.2, random_state=42, stratify=y_over_dumm)

        # Tokenizer using keras 
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(X_train)

        #tok_name = 'tokenizer'
        #tok_save_path = InterMConfig().tokenizer_save_path   
        #joblib.dump(tokenizer, tok_save_path+'['+date_only+hour_only+min_only+']_'+tok_name+'.joblib')
        
        #en_load_path = InterMConfig().encoding_load_path
        #enc = joblib.load(en_load_path)


        joblib.dump(tokenizer,'./weight/tokenizer.joblib')
        tokenizer = joblib.load('./weight/tokenizer.joblib')
        X_train_encoded = tokenizer.texts_to_sequences(X_train)
        word_to_index = tokenizer.word_index
        message = '=' * 25+'<5. Tokenizer end>'+'=' * 25
        logger.info(message)

        # 패딩을 위한 토큰인 0번 단어를 고려하여 +1 하여 저장
        vocab_size = len(word_to_index) + 1
        # padding using keras
        max_len = max(len(sample) for sample in X_train_encoded) # X_trsin Row 기준 인코딩 값중 가장 길이가 긴 값 
        X_train_padded = pad_sequences(X_train_encoded, maxlen = max_len)
        message = '=' * 25+'<6. Padding set end>'+'=' * 25
        logger.info(message)

        # 모델링
        embedding_dim =1 #32
        hidden_units =1 #100

        model = Sequential()
        model.add(Embedding(vocab_size, embedding_dim))
        model.add(Bidirectional(LSTM(hidden_units,dropout=0.2,return_sequences=True)))
        model.add(Bidirectional(LSTM(int(hidden_units/2),dropout=0.2)))
        model.add(Dense(y_train.shape[1], activation='softmax'))

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        callbacks = EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)
        history = model.fit(X_train_padded, y_train, epochs=1, batch_size=64, validation_split=0.2, callbacks=[callbacks])
        #epoch = 40
        message = '=' * 25+'<7. Bi-LSTM model training end>'+'=' * 25
        logger.info(message)

        # 예측 결과 확인
        X_test_encoded = tokenizer.texts_to_sequences(X_test)
        X_test_padded = pad_sequences(X_test_encoded, maxlen = max_len)

        eveluate_run = model.evaluate(X_test_padded, y_test)

        message = '=' * 25+'<8. Bi-LSTM trained model evaluation>'+'=' * 25
        logger.info(message)

        message = '=' * 20+"테스트 Loss: %.4f" % eveluate_run[0]
        logger.info(message)
        message = '=' * 20+"테스트 Accuracy: %.4f" % eveluate_run[1]
        logger.info(message)

        #print("\n 테스트 Loss: %.4f" % eveluate_run[0])
        #print("\n 테스트 Accuracy: %.4f" % eveluate_run[1])

        # 학습 모델 저장
        model_name = 'BiLSTM_ver1'
        model.save('./model/['+date_only+hour_only+min_only+']_'+model_name+'.h5')
        message = '=' * 25+'<9. Bi-LSTM trained model save>'+'=' * 25
        logger.info(message)

        print('U R good Enough')

        print('RUNNING TIME => ',datetime.now() - start)
    