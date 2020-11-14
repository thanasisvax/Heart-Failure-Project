import json
import numpy as np
import os
from sklearn.externals import joblib


def init():
    global model
    # model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'automlmodel1.pkl')
    model_path = os.path.join(os.environ['AZUREML_MODEL_DIR'], 'automlmodel1.pkl')
    model = joblib.load(model_path)

def run(data):

        data = np.array(json.loads(data))
        result = model.predict(data)
        # You can return any data type, as long as it is JSON serializable.
    return result.tolist()

