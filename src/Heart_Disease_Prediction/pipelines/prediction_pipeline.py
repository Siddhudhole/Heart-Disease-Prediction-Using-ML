import os,sys
import pandas as pd
from pathlib import Path 
import numpy as np  
from dataclasses import dataclass 
from src.Heart_Disease_Prediction.utils import load_model 
from src.Heart_Disease_Prediction.logger import logging 


# Configuration setup : model loaded and Processor loaded 

@dataclass
class PredictionConfig:
    model  = load_model(os.path.join(os.getcwd(),os.path.join("artifacts","models/model.pkl")))
    processor = load_model(os.path.join(os.getcwd(),os.path.join("artifacts","models/processor.pkl"))) 

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
            logging.info("Prediction completed successfully")
            return prediction[0] 
        except Exception as e:
            return str(e) 
        
