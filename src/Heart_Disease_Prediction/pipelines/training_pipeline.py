import os,sys 
import pandas as pd 
from datetime import datetime 
import mlflow 
import dagshub 
from pathlib import Path 
from src.Heart_Disease_Prediction.exception import CustomException 
from src.Heart_Disease_Prediction.logger import logging
from src.Heart_Disease_Prediction.utils import evaluate_model 
from src.Heart_Disease_Prediction.components.data_ingestion import DataIngestionConfig,DataIngestion 
from src.Heart_Disease_Prediction.components.data_transformation import DataTransformation,DataTransformationConfig 
from src.Heart_Disease_Prediction.components.model_tranier import ModelTrainer,ModelTrainerConfig
 



if __name__ == "__main__":

    try:
        # Step :Data Ingestion 
        logging.info("-------------Data Ingestion------------------------") 
        data_ingestion = DataIngestion()
        train_data_path, test_data_path = data_ingestion.load_data() 
        logging.info("Data Ingestion is complete successfully  at time {}".format(datetime.now()))
        logging.info("----------------------------------------------------")


        # Step 2: Data Transformation
        logging.info("-------------Data Transformation------------------------")
        data_transformation = DataTransformation()
        train_data,test_data= data_transformation.transform(train_path=train_data_path,test_path=test_data_path)
        logging.info("Data transformation is completed successfully at time{}".format(datetime.now()))
        logging.info("----------------------------------------------------")


        # Step 3 : Model Training 
        logging.info("-------------Model Training------------------------") 
        x_train,y_train = train_data[:,:-1],train_data[:,-1] 
        model_trainer = ModelTrainer()
        model= model_trainer.trainer(x_train,y_train)
        logging.info("Model Training Pipeline is completed at time {}".format(datetime.now()))
        logging.info("----------------------------------------------------") 

        # Step 4 : Model Evaluation 
        logging.info("-------------Model Evaluation------------------------") 
        x_test,y_test = test_data[:,:-1],test_data[:,-1]
        y_pred = model.predict(x_test)
        accuracy, precision, recall=evaluate_model(y_test=y_test,y_pred=y_pred)
        print("accuracy "+ str(accuracy) +" precision "+str(precision),"recall "+ str(recall)) 
        logging.info("Model Evaluation Pipeline is completed at time {}".format(datetime.now()))
        logging.info("----------------------------------------------------") 

         
    

    except Exception as e:
        logging.error("Error transforming "+ str(e))
        raise CustomException(e,sys)