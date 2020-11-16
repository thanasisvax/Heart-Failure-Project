import joblib
import json
import numpy as np
import os
from azureml.core.model import Model
import time

# The init() method is called once, when the web service starts up.
#
# Typically you would deserialize the model file, as shown here using joblib,
# and store it in a global variable so your run() method can access it later.
def init():
    global model
    #Print statement for appinsights custom traces:
    print ("model initialized" + time.strftime("%H:%M:%S"))

    # The AZUREML_MODEL_DIR environment variable indicates
    # a directory containing the model file you registered.
    #model_filename = 'Automl_run.model_id' - Replace model_name with the AutoML ID.
    model_path = Model.get_model_path('AutoMLc543f296e15')

    model = joblib.load(model_path)   

# The run() method is called each time a request is made to the scoring API.
#
# Shown here are the optional input_schema and output_schema decorators
# from the inference-schema pip package. Using these decorators on your
# run() method parses and validates the incoming payload against
# the example input you provide here. This will also generate a Swagger
# API document for your web service.

def run(data):
    try:
        data = np.array(json.loads(data)['data'])
        result = model.predict(data)
        # Log the input and output data to appinsights:
        info = {
            "input": data,
            "output": result.tolist()
            }
        print(json.dumps(info))
        # You can return any data type, as long as it is JSON serializable.
        return result.tolist()
    except Exception as e:
        error = str(e)
        print (error + time.strftime("%H:%M:%S"))
        return error
