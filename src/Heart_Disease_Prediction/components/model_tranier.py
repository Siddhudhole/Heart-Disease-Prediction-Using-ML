import os ,sys 
import mlflow 
import optuna 
import dagshub 
from pathlib import Path 
from dataclasses import dataclass 
from mlflow.models import infer_signature 
from sklearn.metrics import accuracy_score 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier 
from src.Heart_Disease_Prediction.logger import logging
from src.Heart_Disease_Prediction.exception import CustomException 
from src.Heart_Disease_Prediction.utils import save_model 
from src.Heart_Disease_Prediction.utils import evaluate_model 

 

@dataclass 
class ModelTrainerConfig:
    model_path:str = os.path.join('artifacts/models','model.pkl')
    
class ModelTrainer():
    def __init__(self):
        self.config = ModelTrainerConfig()

    def trainer(self,x,y):
        try :
            logging.info("Training model...")
            dagshub.init(repo_owner='Siddhudhole', repo_name='Heart-Disease-Prediction-Using-ML', mlflow=True) 
            logging.info("dagshub initialized successfully") 
            mlflow.set_experiment("Heart_Disease_Prediction")
            mlflow.set_tracking_uri("https://dagshub.com/Siddhudhole/Heart-Disease-Prediction-Using-ML.mlflow")
            with mlflow.start_run() as run:
                x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=0.1, random_state=42)
                def objective(trial):
                            n_estimators = trial.suggest_int('n_estimators',10,100)
                            criterion = trial.suggest_categorical('criterion',choices=['gini','entropy','log_loss'])
                            max_depth = trial.suggest_int('max_depth',1,10) 
                            random_state = 42
                            max_features = trial.suggest_categorical('max_features',choices=['sqrt','log2'])
                            model = RandomForestClassifier(n_estimators=n_estimators, criterion=criterion, 
                                                                max_depth=max_depth, random_state=random_state,
                                                                max_features=max_features)
                            model.fit(x_train,y_train)
                            y_preds = model.predict(x_valid) 
                            accuracy = accuracy_score(y_true=y_valid,y_pred=y_preds)
                            return accuracy
                study = optuna.create_study(direction='maximize')
                study.optimize(objective, n_trials=20)
                logging.info('Best trial: score {}, params: {}'.format(study.best_value, study.best_params))
                model = RandomForestClassifier(n_estimators=study.best_params['n_estimators'], 
                                                            criterion=study.best_params['criterion'], 
                                                                max_depth=study.best_params['max_depth'], 
                                                                random_state=42,
                                                                max_features=study.best_params['max_features'])
                model.fit(x, y)
                y_pred = model.predict(x)
                logging.info("Model preparation completed")
                mlflow.log_param('n_estimators',study.best_params['n_estimators'])
                mlflow.log_param('criterion',study.best_params['criterion'])
                mlflow.log_param('max_depth',study.best_params['max_depth'])
                mlflow.log_param('max_features',study.best_params['max_features'])
                accuracy, precision, recall  = evaluate_model(y_test=y,y_pred=y_pred)
                mlflow.log_metric("accuracy", accuracy)
                mlflow.log_metric("precision", precision)
                mlflow.log_metric("recall", recall)
                save_model(model,self.config.model_path)
                logging.info('Model save successfully')
                return model 
        except Exception as e:
            logging.error('Error getting model trainer file'+str(e))
            raise CustomException(e,sys)