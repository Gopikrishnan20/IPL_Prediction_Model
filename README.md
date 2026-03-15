# IPL_Prediction_Model
Overview

This project builds a Machine Learning model to predict the winner of Indian Premier League (IPL) cricket matches using historical IPL data.
The model uses basic match-level features such as teams, toss result, venue, and city to predict the probable match winner.

This is the base version (V1) of the project and serves as a foundation for future improvements such as feature engineering, advanced models, and live match prediction.

Dataset

The dataset used is a ball-by-ball IPL dataset containing detailed information about every delivery in IPL matches.

Since the model predicts match outcomes, the dataset is first converted into match-level data, where each match is represented by a single row with relevant match features.

Features Used

Team 1

Team 2

Toss Winner

Toss Decision

Venue

City

Target Variable

Match Winner

Machine Learning Pipeline

The project follows the following pipeline:

Load IPL dataset

Convert ball-by-ball data to match-level data

Clean dataset and remove missing values

Encode categorical features using LabelEncoder

Split dataset into:

50% Training data

20% Validation data

30% Testing data

Train a Random Forest Classifier

Evaluate model performance

Save the trained model for future predictions

Model

Algorithm used:

Random Forest Classifier

Random Forest was chosen because it performs well with categorical data and reduces overfitting by combining multiple decision trees.
