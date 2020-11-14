import json
import numpy as np
import os
from sklearn.externals import joblib


def init():
    print("init method has been Initiated")    
    global model
    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'automlmodel.pkl')
    #model = joblib.load(model_path)
    model = load_model(model_path)
    print("init method has been completed")
def run(data):
    try:
        data = np.array(json.loads(data))
        result = model.predict(data)
        # You can return any data type, as long as it is JSON serializable.
        return result.tolist()
    except Exception as e:
        error = str(e)
        return error
