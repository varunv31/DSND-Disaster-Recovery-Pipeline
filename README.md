# DSND-Disaster-Recovery-Pipeline
A Udacity project that helps to build the ETL and ML Pipeline

# Project Overview
This repository that is created for all of us to understand on how to create ETL and ML pipelines. Along wtih detailed pipelines, it contains a web app which an emergency worker could use during a disaster event.

# Structure
Primarily contains the following folders
1. data: This hosts the ETL code for preparing the clean data and saving the Disaster_Combined file in the DisasterResponse.db
2. models: This hosts the ML code for training and testing the model. Currently have used RandomForest to create the model with optimized GridSearch
3. app: This hosts the code to create the flask web app usin the classifier.pkl

# Run process_data.py
Save the data folder in the current working directory and process_data.py in the data folder.
From the current working directory, run the following command: python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db

# Run train_classifier.py
In the current working directory, create a folder called 'models' and save train_classifier.py in this.
From the current working directory, run the following command: python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl

# Run the web app
Save the app folder in the current working directory.
Run the following command in the app directory: python run.py
Go to http://0.0.0.0:3001/
