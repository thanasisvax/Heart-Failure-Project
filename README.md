# Heart Failure Project

## Overview
On this project, I have used the Heart Failure dataset from Kaggle to predict the possibility of a patient to have a heart failure based on his/her medical records. This is a classification problem. I have used two different train methods to achieve the best score. One training method is using the hyperdrive to optimize the hyperparameters of the logistic regression model for the heart failure dataset. The second training method was the utilization of the azure automl to find the best model fit for the heart failure dataset.</p>
Hyperdrive Model -> hyperparameter_tuning.ipynb </p>
AutoMl Model -> automl.ipynb </p>
Finally, the best obtained model from the above two training methods is deployed using a webservice in the Azure ML studio.

## Project Set Up and Installation
This project requires access into the Azure ML studio. The following steps to be followed to initialize the project:</p>
1. Create a new dataset in the Azure ML studio importing the local heart_failure_clinical_records_dataset.csv file which is attached in this repository.</p>
2. Import all the notebooks attached in this repository and the test-data.csv file in the Notebooks section in Azure ML studio.</p>
3. Create a new compute target in the Azure ML studio and run the above notebooks using jupyter notebooks.</p>
4. All instructions are detailed in the notebooks.

## Dataset

### Overview
The Heart Fialure dataset was obtained from the Kaggle website. https://www.kaggle.com/andrewmvd/heart-failure-clinical-data </p>
There are 12 features as clinical records of the patients and the prediction is on the DEATH_EVENT columnn.
### Task
All of the twelve features are used from this dataset. The DEATH_EVENT is the prediction for our classification problem.</p>
For the hyperdrive, I have used the SKLEARN logistic regression algorithm to predict the DEATH_EVENT column using the heart failure dataset.</p>
From the automl run becomes apparent that the best fitted model is the Voting Ensemble model. 

### Access
I have downloaded the dataset from the Kaggle website. Then, I have registered the dataset in the Azure ML studio. The data format is tabular and this is accepted for the automl run.</p>
In the notebook, I have used the following command to import the dataset from my workspace - ds = Dataset.get_by_name(ws, name='Heart-Failure').

## Automated ML
The most important settings for the automl run are the following:</p>
The task is classification.</p>
I split the dataset into training and test data and then fed into the automl both of them.</p>
The prediction label is set as DEATH_EVENT.</p>
The primary metric is the accuracy.</p>
Experiment timeouts after 30 mins.</p>
featurization is set to auto.

### Results
The voting Ensemble algorithm is the best fitted model with accuracy 0.91. The parameters of the automl are detailed in the previous section and also shown in the picture 1 below.</p>
![alt](https://github.com/thanasisvax/Heart-Failure-Project/blob/master/starter_file/Automl%20settings.PNG)</p>

One of the impovements will be to collect more data for this classification problem. Also, it might be worthy to test the automl using deep learing models as well to see if the accuracy is increased. 

Screenshots of the RunDetails are shown below:</p>
![alt](https://github.com/thanasisvax/Heart-Failure-Project/blob/master/starter_file/RunDetails_AutoML.PNG)</p>
![alt](https://github.com/thanasisvax/Heart-Failure-Project/blob/master/starter_file/RunDetails_AutoML_2.PNG)</p>

Screenshot of the best model is shown below:</p>
![alt](https://github.com/thanasisvax/Heart-Failure-Project/blob/master/starter_file/Best%20Model_Automl.PNG)</p>

## Hyperparameter Tuning
For that training model, I used logistic regression because the task for the machine learning algorithm was to solve a classification problem. For that reason, I used the SKLEARN logistic regression model to predict the DEATH_EVENT. I tried to optimize the "C" and "max_iter" parameters of the logistic regression algorithm. The C parameter represents the inverse regularization strenght and the max_iter parameter represents the maximum number of iterations for the solvers to converge. The "C" parameter was a selection of numbers in uniform from 0.1 to 1. For the "max_iter" parameter was a choise from the following number of iterations (5, 10,  20,  40, 50, 80, 100, 150, 200).

### Results
The logistic regression model achieved an accuracy of 0.78. The parameters of achieving that accuracy was "C":0.26 and "max_iter": 40 as shown in the picture below:</p>
Screenshot of the best model:</p>
![alt](https://github.com/thanasisvax/Heart-Failure-Project/blob/master/starter_file/Best%20model%20Hyperdrive.PNG)</p>

Screenshot of the RunDetails is shown below:</p>
![alt](https://github.com/thanasisvax/Heart-Failure-Project/blob/master/starter_file/Hyperdrive_RunDetails.PNG)</p>

## Model Deployment
The model with the best accuracy is the voting Ensemble model from the automl run. Initially, I registered the model in the workspace. Then, I created the environment for my deployment to run. After that, I created the score.py file which is attached in this repository with name: scope.py. Then, I deployed locally and then using as webservice my registered model. Finally, the webservice was tested with inputs from the test-data.csv file.</p>
As it can be seen in the notebook, I sued the json command in order to send the data in the webservice and from then being processed from the score.py file to receive back a response of prediction. </p>

Screenshot is shown the response from my webservice:</p>
![alt](https://github.com/thanasisvax/Heart-Failure-Project/blob/master/starter_file/Webservice%20Outcome.PNG)</p>

## Screen Recording
Screen recording is uploaded in the following link:

## Standout Suggestions
I have enabled the application insights which it can be seen the successful requests and responses of my model. see screenshot below:</p>
![alt](https://github.com/thanasisvax/Heart-Failure-Project/blob/master/starter_file/Applications%20Insights.PNG)</p>

Improvements for my model is as such:</p>
1. collect more data in order to avoid overfitting.</p>
2. Try deep-learning methods in order to see if this achieves better results.
