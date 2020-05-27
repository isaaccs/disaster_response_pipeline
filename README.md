# Disaster Response Ripeline

## Table of contents
* [Motivation](#Motivation)
* [Instructions](#Instructions)
* [Project Components](#Project_Components)
* [Technologies](#technologies)
* [Contact](#Contact)

## Motivations:
In this Workspace, you'll find a data set containing real messages that were sent during disaster events. A machine learning pipeline is create to categorize these events so that you can send the messages to an appropriate disaster relief agency.

## Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

## Project Components
There are three components you'll need to complete for this project.

### ETL Pipeline
In a Python script, process_data.py, write a data cleaning pipeline that:

* Loads the messages and categories datasets
* Merges the two datasets
* Cleans the data
* Stores it in a SQLite database

### ML Pipeline
In a Python script, train_classifier.py, write a machine learning pipeline that:

* Loads data from the SQLite database
* Splits the dataset into training and test sets
* Builds a text processing and machine learning pipeline
* Trains and tunes a model using GridSearchCV
* Outputs results on the test set
* Exports the final model as a pickle file

### Flask Web App
Data visualizations using Plotly in the web app. 

## Technologies
### Languages
Project is created with Python 3.6.9.

### Dependencies


* [NumPy](https://numpy.org)
* [Matplotlib](https://matplotlib.org)
* [pandas](https://pandas.pydata.org)
* [NLTK](https://www.nltk.org/)
* [joblib]
* [sqlalchemy]
* [string]
* [flask]
* [plot.ly]





## Contact

* Mail: isaaccohensabban_at_gmail_dot_com

