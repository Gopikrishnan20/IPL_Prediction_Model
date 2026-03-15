import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

df = pd.read_csv("IPL.csv")

print("Dataset Loaded")
print(df.head())


match_df = df[['match_id',
               'batting_team',
               'bowling_team',
               'toss_winner',
               'toss_decision',
               'venue',
               'city',
               'match_won_by']]

match_df = match_df.drop_duplicates(subset='match_id')

match_df.rename(columns={
    'batting_team': 'team1',
    'bowling_team': 'team2',
    'match_won_by': 'winner'
}, inplace=True)

match_df = match_df.dropna(subset=['winner'])

print("\nCleaned Match Dataset")
print(match_df.head())

encoder_dict = {}

categorical_columns = [
    'team1',
    'team2',
    'toss_winner',
    'toss_decision',
    'venue',
    'city',
    'winner'
]

for col in categorical_columns:
    le = LabelEncoder()
    match_df[col] = le.fit_transform(match_df[col])
    encoder_dict[col] = le


X = match_df.drop(columns=['winner', 'match_id'])
y = match_df['winner']

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y,
    test_size=0.5,
    random_state=42
)

X_test, X_val, y_test, y_val = train_test_split(
    X_temp,
    y_temp,
    test_size=0.4,
    random_state=42
)

print("\nDataset Split")
print("Training size:", len(X_train))
print("Testing size:", len(X_test))
print("Validation size:", len(X_val))

model = RandomForestClassifier(
    n_estimators=300,
    random_state=42
)

model.fit(X_train, y_train)

print("\nModel Trained")

val_pred = model.predict(X_val)

val_accuracy = accuracy_score(y_val, val_pred)

print("Validation Accuracy:", val_accuracy)


test_pred = model.predict(X_test)

test_accuracy = accuracy_score(y_test, test_pred)

print("Test Accuracy:", test_accuracy)


pickle.dump(model, open("ipl_winner_model.pkl", "wb"))

print("\nModel saved as ipl_winner_model.pkl")


def predict_winner():

    print("\nEnter Match Details\n")

    team1 = input("Team 1: ")
    team2 = input("Team 2: ")
    toss_winner = input("Toss Winner: ")
    toss_decision = input("Toss Decision (bat/field): ")
    venue = input("Venue: ")
    city = input("City: ")

    team1 = encoder_dict['team1'].transform([team1])[0]
    team2 = encoder_dict['team2'].transform([team2])[0]
    toss_winner = encoder_dict['toss_winner'].transform([toss_winner])[0]
    toss_decision = encoder_dict['toss_decision'].transform([toss_decision])[0]
    venue = encoder_dict['venue'].transform([venue])[0]
    city = encoder_dict['city'].transform([city])[0]

    input_data = np.array([[team1, team2, toss_winner,
                            toss_decision, venue, city]])

    prediction = model.predict(input_data)

    winner = encoder_dict['winner'].inverse_transform(prediction)

    print("\nPredicted Match Winner:", winner[0])

predict_winner()