import os
import sys
import logging
import numpy as np
from dataclasses import dataclass
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from imblearn.combine import SMOTEENN
from src.exception import CustomException
from src.utils import save_object
import pandas as pd

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')
    preprocessor_file_path = os.path.join('artifacts', 'preprocessor.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Eğitim ve test verilerinden bağımlı ve bağımsız değişkenleri ayırma")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )

            # Hedef değişkeni sayısal formata dönüştür
            y_train = np.where(y_train == 'Yes', 1, 0)
            y_test = np.where(y_test == 'Yes', 1, 0)

            # numpy dizilerini pandas DataFrame'e dönüştür
            feature_names = ['gender', 'SeniorCitizen', 'partner', 'tenure', 'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod', 'MonthlyCharges', 'TotalCharges', 'feature19', 'feature20', 'feature21', 'feature22', 'feature23', 'feature24', 'feature25', 'feature26', 'feature27', 'feature28', 'feature29', 'feature30', 'feature31', 'feature32', 'feature33', 'feature34', 'feature35', 'feature36', 'feature37', 'feature38', 'feature39', 'feature40', 'feature41', 'feature42']
            X_train_df = pd.DataFrame(X_train, columns=feature_names)
            X_test_df = pd.DataFrame(X_test, columns=feature_names)

            # pandas DataFrame'leri tekrar numpy dizilerine dönüştür
            X_train = X_train_df.values
            X_test = X_test_df.values

            features_to_drop = ['feature19', 'feature20', 'feature21', 'feature22', 'feature23', 'feature24', 'feature25', 'feature26', 'feature27', 'feature28', 'feature29', 'feature30', 'feature31', 'feature32', 'feature33', 'feature34', 'feature35', 'feature36', 'feature37', 'feature38', 'feature39', 'feature40', 'feature41', 'feature42']
            X_train_df = X_train_df.drop(columns=features_to_drop)
            X_test_df = X_test_df.drop(columns=features_to_drop)

            # X_train ve X_test değerlerini pandas DataFrame olarak yazdır
            print("X_train DataFrame:")
            print(X_train_df.head())
            print("X_test DataFrame:")
            print(X_test_df.head())

            # SMOTEENN uygulama
            smote_enn = SMOTEENN(random_state=42)
            X_train_resampled, y_train_resampled = smote_enn.fit_resample(X_train, y_train)

            # y_train sınıf dağılımı (SMOTEENN sonrası)
            print("y_train sınıf dağılımı (SMOTEENN sonrası):")
            print(pd.Series(y_train_resampled).value_counts())

            print("y_train_resampled:")
            print(y_train_resampled)

            models = {
                "Random Forest": RandomForestClassifier(class_weight='balanced'),
                "Decision Tree": DecisionTreeClassifier(class_weight='balanced'),
                "Gradient Boosting": GradientBoostingClassifier()
            }

            best_model = None
            best_accuracy = 0

            # Modelleri eğit ve değerlendir
            for model_name, model in models.items():
                logging.info(f"{model_name} eğitiliyor")
                model.fit(X_train_resampled, y_train_resampled)
                predictions = model.predict(X_test)
                accuracy = accuracy_score(y_test, predictions)
                f1 = f1_score(y_test, predictions)
                precision = precision_score(y_test, predictions)
                recall = recall_score(y_test, predictions)
                logging.info(f"{model_name} Doğruluk: {accuracy}, F1 Skoru: {f1}, Precision: {precision}, Recall: {recall}")

                if f1 > best_accuracy:  # F1 skoruna göre en iyi modeli seç
                    best_accuracy = f1
                    best_model = model

            # En iyi modeli kaydet
            if best_model is not None:
                save_object(self.model_trainer_config.trained_model_file_path, best_model)
                logging.info(f"En iyi model kaydedildi: {best_model.__class__.__name__} F1 Skoru: {best_accuracy}")

        except Exception as e:
            raise CustomException(e, sys)

    def train_model(self, X_train, y_train, X_test, y_test, models):
        model_report = {}
        for model_name, model in models.items():
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            score = accuracy_score(y_test, predictions)
            model_report[model_name] = {'score': score}
        return model_report