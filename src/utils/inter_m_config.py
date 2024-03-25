import configparser


class InterMConfig:
    def __init__(self):
        self._config = configparser.ConfigParser()
        self._config.read('./config/config.ini')
        self.train_data_path = self._config['FIRST_TRAIN_DATA']['train_data_path']

        self.embedding_dimention = self._config['MODEL_PARAMETER']['embeddig_dimention']
        self.hidden_units = self._config['MODEL_PARAMETER']['hidden_units']
        self.drop_out = self._config['MODEL_PARAMETER']['drop_out']
        self.epochs = self._config['MODEL_PARAMETER']['epochs']

        self.model_save_path = self._config['MODEL_SAVE']['model_save_path']
        self.model_load_path = self._config['MODEL_LOAD']['model_load_path']
        
        self.encoding_save_path = self._config['ONE_HOT_ENCODING_SAVE']['encoding_save_path']
        self.encoding_load_path = self._config['ONE_HOT_ENCODING_LOAD']['encoding_load_path']        
        
        self.tokenizer_save_path = self._config['TOKENIZER_SAVE']['tokenizer_save_path']
        self.tokenizer_load_path = self._config['TOKENIZER_LOAD']['tokenizer_load_path']       

        #self. = self._config['']['']
