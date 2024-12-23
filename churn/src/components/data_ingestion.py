import os
import sys
from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

from src.exception import CustomException
from src.logger import logging

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig
from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer



@dataclass # __init__ fonksiyonunu yazmadan özellik tanımamı sağlar
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts',"train.csv")
    test_data_path: str = os.path.join('artifacts',"test.csv")
    raw_data_path: str = os.path.join('artifacts',"data.csv")


    
class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig() #train, test ve ham datanın yollarını oluşturma işini ingestion_config değişkenine atıyor.

    def initiate_data_ingestion(self):
        logging.info('Data İngestion Sınıfına Girildi ve Çalışıldı')
        try:
            df=pd.read_csv('data/Churn.csv')
            logging.info('Veri Seti Okundu')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path)) # train_data nın klasörünü oluşturur

            df.to_csv(self.ingestion_config.raw_data_path, index=False,header = True) # Ham datayı csv formatında bir dosyaya dönüştürerek kaydeder

            logging.info('Train Test Data Ayrımına Başlandı')
            train_set, test_set = train_test_split(df, test_size = 0.2, random_state = 42)

            train_set.to_csv(self.ingestion_config.train_data_path, index = False, header = True)

            test_set.to_csv(self.ingestion_config.test_data_path, index = False, header = True)
            
            logging.info('Data ayırma işlemi başarıyla tamamlandı')

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e, sys)
        
if __name__ == '__main__':
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()

    
    data_transformation = DataTransformation()
    train_arr,test_arr,_ = data_transformation.initiate_data_transformation(train_data, test_data)

    modeltrainer = ModelTrainer()
    print(modeltrainer.initiate_model_trainer(train_arr,test_arr))


