import os,sys
import pandas as pd
from pathlib import Path 
import numpy as np  
from dataclasses import dataclass 
from src.Heart_Disease_Prediction.logger import logging 
from src.Heart_Disease_Prediction.utils import load_model 
from src.Heart_Disease_Prediction.exception import CustomException 


# Configuration setup : model loaded and Processor loaded 

@dataclass
class PredictionConfig:
    model = load_model(Path("artifacts\models\model.pkl"))
    processor = load_model(Path("artifacts\models\processor.pkl")) 

class Prediction:
    def __init__(self):
        self.model_config =PredictionConfig()
    
    def predict(self,data:list):

        try:
            logging.info("Predicting...") 
            data = np.array(data) 
            data = data.reshape(1,-1)
            df = pd.DataFrame(data=data,columns=['age','gender','impluse','pressurehight','pressurelow','glucose','kcm','troponin'])
            df = self.model_config.processor.transform(df)
            prediction = self.model_config.model.predict(df)
            logging.info("Prediction completed successfully ")
            return prediction[0] 
        except Exception as e:
            raise CustomException(e,sys)
        

# Usage: 
if __name__ == '__main__':
    prediction = Prediction()
    data = np.array([64,1,66,160,83,160,1.8,0.012])
    prediction_result = prediction.predict(data)
    print(f"Predicted Heart Disease: {prediction_result}")
