import os
import sys

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import numpy as np 
import pandas as pd
import pickle
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_model(X_train, y_train, X_test, y_test, models, param):
    try:
        report = {}
        for i in range(len(models)):
            model = list(models.values())[i]
            para = param[list(models.keys())[i]]
            gs = GridSearchCV(model, para, cv=3, scoring='accuracy')

            gs.fit(X_train, y_train)  # Hiperparametre araması yap
            best_model = gs.best_estimator_  # En iyi modeli al
            best_model.fit(X_train, y_train)  # En iyi model ile eğitim yap

            y_test_pred = best_model.predict(X_test)  # Test verisi üzerinde tahmin yap

            # Performans metriklerini hesapla
            test_accuracy = accuracy_score(y_test, y_test_pred)
            test_f1 = f1_score(y_test, y_test_pred, average='weighted')
            if hasattr(best_model, "predict_proba"):
                test_roc_auc = roc_auc_score(y_test, best_model.predict_proba(X_test)[:, 1])
            else:
                test_roc_auc = None

            report[list(models.keys())[i]] = {
                'accuracy': test_accuracy,
                'f1_score': test_f1,
                'roc_auc': test_roc_auc
            }

        return report

    except Exception as e:
        raise CustomException(e, sys)

def load_object(file_path):
    try:
        with open(file_path, 'rb') as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)
    