## Labor Prediction Model

For glass installation companies, accurately esimtating how long an installation will take to complete is critical to pricing a job. This model predicts how long a glass installation will take.

## Project Set Up and Installation
To run this project, download the zip file from Github and extract the files. Launch the Azure ML Studio and create a new dataset that points to the local csv file from the zip file that you downloaded. Then go to Notebooks and upload the whole folder so that you have all the files accessible. You can then run the automl.ipynb and hyperparameter_tuning.ipynb files.

## Dataset

### Overview
The data is culled from a real glass company's database and extensive feature engineering was done to achieve optimum results. 

### Task
The model predicts how long a job will take based on the following features:
Glass height - how tall the glass is
Glass width - how wide the glass is
Frame material - what the frame holding the glass is made out of
Frame function - what the frame holding the glass actually is (ex. slider, fixed, sash....)
Lift - how high up the glass is
Set - what material is used to keep the glass in the frame
Location type - what kind of place the installation is taking place in (ex. residential, hospital, school...)
Lead Mechanic - who did the installation
Long Carry - whether they had to carry the glass over a long distance

### Access
The data is stored as a csv file in Github. For the purposes of this project, I downloaded the whole Github file to the computer that I was working on and then created a dataset in Azure that referenced the csv file on the local computer.

## Automated ML
First, I cleaned the dataset, which entailed using one-hot-encoding for the categorical columns and normalizing the numeric columns. Then I split the dataset 80/20 into train and test sets. Finally, I created an AutoMLConfig file with the task set as "regression", the primary metric as "r2 score", and the number of cross validations as 10. I also set the experiment timeout to be 30 minutes because of limitations in the time allowed for the Azure workspace. 

### Results
The best model used the VotingEnsemble algorithm and achieved an r2 score of .30299. The parameters used were as follows:
PreFittedSoftVotingRegressor(estimators=[('20', Pipeline(memory=None, steps=[('robustscaler', RobustScaler(copy=True, quantile_range=[10, 90], with_centering=True, with_scaling=False)), ('elasticnet', ElasticNet(alpha=0.001, copy_X=True, fit_intercept=True, l1_ratio=0.21842105263157896, max_iter=1000, normalize=False, positive=False, precompute=False, random_state=... reg_lambda=0.7291666666666667, scale_pos_weight=1, seed=None, silent=None, subsample=0.9, tree_method='hist', verbose=-10, verbosity=0))], verbose=False))], weights=[0.06666666666666667, 0.06666666666666667, 0.06666666666666667, 0.06666666666666667, 0.06666666666666667, 0.2, 0.06666666666666667, 0.06666666666666667, 0.13333333333333333, 0.06666666666666667, 0.13333333333333333]))]

I think that the model could be improved by running the AutoML for a longer period of time, and also maybe by experimenting witht he number of cross validations.

Following are screenshots from the RunDetails widget as well as the best model trained with its parameters:

![AutoML Run Details](https://github.com/SLane35/nd00333-capstone/blob/master/starter_file/AutoML%20Run%20Details.png)
![AutoML While Running](https://github.com/SLane35/nd00333-capstone/blob/master/starter_file/AutoML%20While%20Running.png)
![AutoML Best Model](https://github.com/SLane35/nd00333-capstone/blob/master/starter_file/AutoML%20Best%20Model.png)
![AutoML Best Model Parameters 1](https://github.com/SLane35/nd00333-capstone/blob/master/starter_file/AutoML%20Parameters1.png)
![AutoML Best Model Parameters 2](https://github.com/SLane35/nd00333-capstone/blob/master/starter_file/AutoML%20Parameters2.png)

## Hyperparameter Tuning
I chose a Linear Regression for this experiment since that most closely fit the data that I was using. Since I am trying to estimate how long a job will take to complete, this required a regression algorithm and Linear Regression seemed like the best fit. I used the BayesianParameterSampling for my two parameters: fit intercept and normalize. Both of these can either be 0 or 1. 

I set the primary metric to r2 in order to be able to compare to the AutoML run.

### Results
The best model used the VotingEnsemble algorithm and achieved an r2 score of .34497. The parameters used were as follows: fit intercept: 0, normalize: 0. I would like to try experimenting with different parameters to see if I can get better results. 

Following are screenshots from the RunDetails widget as well as the best model trained with its parameters:

![Hyperdrive Run Details](https://github.com/SLane35/nd00333-capstone/blob/master/starter_file/Hyperdrive%20Run%20Details.png)
![Hyperdrive Run Details 2](https://github.com/SLane35/nd00333-capstone/blob/master/starter_file/Hyperdrive%20Run%20Details%202.png)
![Hyperdrive Best Model](https://github.com/SLane35/nd00333-capstone/blob/master/starter_file/Hyperdrive%20Best%20Run.png)
![Hyperdrive Best Model Parameters](https://github.com/SLane35/nd00333-capstone/blob/master/starter_file/Hyperdrive%20Best%20Model%20Parameters.png)

## Model Deployment
In the train.py file, I use joblib.dump to save the model. Then I register the best model after the hyperdrive run has completed. I then create an environment that includes all the necessary conda and pip dependencies and I use that and my score.py file to create an InferenceConfig. The score.py file uses joblib.load to load the model, and then runs the input data through the model and returns the prediction as an output. 

I used ACIWebservice to deploy my model, with auth_enabled=True and enable_app_insights=True. I then deployed my model to the ACI service. Once the model was deployed and the service was running, I was able to query the service using the key in the header for authorization and the scoring uri. I made sure to create the input string as a JSON dictionary which then got converted to a DataFrame in the score.py file. 

## Screen Recording
https://www.loom.com/share/8a3c2512b43942f48fa92bb65bed09f2

