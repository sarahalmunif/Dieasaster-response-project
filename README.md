# Dieasaster-response-project
Udacity Nanodegree Project Disaster Response

This is the project for Udacity Data Engineering program. In this project I used the skiils I learned to analyze disaster data from Figure Eight and build ML model that classify disaster messages. 
In this project I used a real messages data that were send during disasters.

This project will help emergency worker to classify new messages,It includes web app where an emergency worker can input a message and get the classification of the message. 

The project consists of three main parts:
data processing part
training the model part
web app that use the trained model to predict.

You can find below the commands you need to run the modules:

Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Go to `app` directory: `cd app`

3. Run your web app: `python run.py`

"process_data.py": This is python script for data processing.to run the file:


disaster_messages.csv: CSV data file to process
disaster_categories.csv: CSV data file to process
DisasterResponse.db: sql lite database to save cleaned data

"train_classifier.py": This is python script for model training. to run the file:


DisasterResponse.db: sql lite database that store training and testing data.
classifier.pkl: trained model

"run.py": This is python script that starts web app.to run the file:


"master.html": html landing page of web app. Displays the following charts :
Frequency of output categories
Distribution of message genres

"go.hmtl": html page that displays classification results of model.
