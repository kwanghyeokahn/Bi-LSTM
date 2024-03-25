import logging
from logging.handlers import RotatingFileHandler
from src.bi_lstm_train import BiLstmTrain

# log
## 로그 이름
logger = logging.getLogger('inter_m') 
## 디버그 종류에 대한 단위 선택 (info 이상은 다 나와라)
logger.setLevel(logging.INFO)
## 로그파일 경로
log_path = "./log/inter_m_train_log.log"
## 로그파일에 계속 쌓이면, 지정된 용량마다 새로 만들어라 (약 300MB)
file_handler = RotatingFileHandler(log_path, maxBytes=1024*10*100, backupCount=30)
## 아래 포맷으로 로그에 찍혀라
file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logger.addHandler(file_handler)
## 화면에 출력되는 부분 (ex:터미널)
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logger.addHandler(stream_handler)

def main():
    obj = BiLstmTrain()
    obj.run()

if __name__=='__main__':
    main()