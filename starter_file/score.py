import json
import numpy as np
import os
import joblib
from pandas import DataFrame

def init():
    global model
    #model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'automlmodel.pkl')
    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'bestmodel.pkl')
    model = joblib.load(model_path)

def run(data):
    try:
        data = json.loads(data)
        test_data = data["data"]
        data = DataFrame(test_data)
        #data = np.array(json.loads(data))
        result = model.predict(data)
        # You can return any data type, as long as it is JSON serializable.
        return result.tolist()
    except Exception as e:
        error = str(e)
        return error