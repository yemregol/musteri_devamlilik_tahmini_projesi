import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object
import os

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = os.path.join('artifacts', 'model.pkl')
            preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')
            print('Yükleme Öncesi')
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            print('Yükleme Sonrası')
            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            probs = model.predict_proba(data_scaled)[:, 1]
            return preds, probs
        except Exception as e:
            raise CustomException(e, sys)

class CustomData:
    def __init__(self, gender, SeniorCitizen, Partner, Dependents, PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies, Contract, PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges, tenure):
        self.gender = gender
        self.SeniorCitizen = SeniorCitizen
        self.Partner = Partner
        self.Dependents = Dependents
        self.PhoneService = PhoneService
        self.MultipleLines = MultipleLines
        self.InternetService = InternetService
        self.OnlineSecurity = OnlineSecurity
        self.OnlineBackup = OnlineBackup
        self.DeviceProtection = DeviceProtection
        self.TechSupport = TechSupport
        self.StreamingTV = StreamingTV
        self.StreamingMovies = StreamingMovies
        self.Contract = Contract
        self.PaperlessBilling = PaperlessBilling
        self.PaymentMethod = PaymentMethod
        self.MonthlyCharges = MonthlyCharges
        self.TotalCharges = TotalCharges
        self.tenure = tenure

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                'gender': [self.gender],
                'SeniorCitizen': [self.SeniorCitizen],
                'Partner': [self.Partner],
                'Dependents': [self.Dependents],
                'PhoneService': [self.PhoneService],
                'MultipleLines': [self.MultipleLines],
                'InternetService': [self.InternetService],
                'OnlineSecurity': [self.OnlineSecurity],
                'OnlineBackup': [self.OnlineBackup],
                'DeviceProtection': [self.DeviceProtection],
                'TechSupport': [self.TechSupport],
                'StreamingTV': [self.StreamingTV],
                'StreamingMovies': [self.StreamingMovies],
                'Contract': [self.Contract],
                'PaperlessBilling': [self.PaperlessBilling],
                'PaymentMethod': [self.PaymentMethod],
                'MonthlyCharges': [self.MonthlyCharges],
                'TotalCharges': [self.TotalCharges],
                'tenure': [self.tenure]
            }

            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            raise CustomException(e, sys)