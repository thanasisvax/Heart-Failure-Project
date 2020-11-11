from sklearn.linear_model import LogisticRegression
import argparse
import os
import numpy as np
from sklearn.metrics import mean_squared_error
import joblib
from sklearn.model_selection import train_test_split
import pandas as pd
from azureml.core.run import Run
from azureml.core.dataset import Dataset
from azureml.core import Experiment, Workspace
from azureml.data.dataset_factory import TabularDatasetFactory

# Load dataset into data variable:
ws = Workspace.from_config()
ds = Dataset.get_by_name(ws, name='Heart-Failure')
data = dataset.to_pandas_dataframe()


#Split Target and Features in y and x respectively in the clean_data function

def clean_data(data):

    y = data['DEATH_EVENT']
    x = data.drop(['DEATH_EVENT'], axis = 1)
    return x ,y

x, y = clean_data(data)

# Split data into train and test sets.

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20, random_state = 42)


run = Run.get_context()


def main():
    # Add arguments to script
    parser = argparse.ArgumentParser()

    parser.add_argument('--C', type=float, default=1.0, help="Inverse of regularization strength. Smaller values cause stronger regularization")
    parser.add_argument('--max_iter', type=int, default=100, help="Maximum number of iterations to converge")

    args = parser.parse_args()

    run.log("Regularization Strength:", np.float(args.C))
    run.log("Max iterations:", np.int(args.max_iter))

    model = LogisticRegression(C=args.C, max_iter=args.max_iter).fit(x_train, y_train)

    accuracy = model.score(x_test, y_test)
    run.log("Accuracy", np.float(accuracy))
    
    os.makedirs('outputs', exist_ok=True)
    # files saved in the "outputs" folder are automatically uploaded into run history
    joblib.dump(model, 'outputs/model.joblib')

if __name__ == '__main__':
    main()


