import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def load_raw_data(path_to_csv):
    return pd.read_csv(path_to_csv)

def preprocess_gender_data(df):
    df.loc[df['Sex'] == 'female', 'Sex'] = 0.0
    df.loc[df['Sex'] == 'male', 'Sex'] = 1.0
    df['Sex'] = df['Sex'].apply(pd.to_numeric)
    return df

def preprocess_cabin_data(df):
    # Keep only the first letter from the cabin data
    df['Cabin'] = [x[0] if not pd.isna(x) else 0 for x in df['Cabin']]
    # Make the letters numeric (A->1, B->2, ...,  no entry->0)
    df['Cabin'] = [ord(x) - 64 if not x == 0 else 0 for x in df['Cabin']]
    df['Cabin'] = df['Cabin'].apply(pd.to_numeric)
    return df

def preprocess_embarked_data(df):
    ports = {'C': 0, 'Q': 1, 'S': 2}
    df['Embarked'] = [ports[x] if x in ports else len(ports) for x in df['Embarked']]
    df['Embarked'] = df['Embarked'].apply(pd.to_numeric)
    return df

def preprocess_age_data(df):
    # Replace NaNs in Age column by the average age
    df.loc[pd.isna(df['Age']), 'Age'] = df['Age'].mean()
    return df

def preprocess_fare_data(df):
    # Replace NaNs in Fare column by the average fare for the class
    for index, row in df.iterrows():
        if pd.isna(row['Fare']):
            average_fare_for_class = np.mean([x['Fare'] for i, x in df.iterrows() if x['Pclass'] == row['Pclass'] and not pd.isna(x['Fare'])])
            df.loc[index, 'Fare'] = average_fare_for_class
    return df

def split_features_target(df):
    target = df['Survived']
    features = df.drop('Survived', axis='columns')
    return features, target

def split_ids_features(df):
    passenger_ids = df['PassengerId']
    features = df.drop('PassengerId', axis='columns')
    return passenger_ids, features

def scale_features(features):
    scaler = MinMaxScaler()
    scaler.fit(features)
    scaled_features = scaler.transform(features)
    scaled_features_df = pd.DataFrame(data=scaled_features, columns=features.columns)
    return scaled_features_df, scaler

def preprocess_columns(df):
    # Preprocess columns that require preprocessing
    df = preprocess_gender_data(df)
    df = preprocess_cabin_data(df)
    df = preprocess_embarked_data(df)
    df = preprocess_age_data(df)
    df = preprocess_fare_data(df)
    return df

def get_train_data(path_to_csv):
    df = load_raw_data(path_to_csv)
    df = df.drop(['PassengerId', 'Name', 'Ticket'], axis='columns')
    df = preprocess_columns(df)

    features, target = split_features_target(df)
    scaled_features, scaler = scale_features(features)
    return scaled_features, target, scaler

def get_test_data(path_to_csv, scaler):
    df = load_raw_data(path_to_csv)
    df = df.drop(['Name', 'Ticket'], axis='columns')
    df = preprocess_columns(df)

    passenger_ids, features = split_ids_features(df)
    scaled_features = scaler.transform(features)
    scaled_features_df = pd.DataFrame(data=scaled_features, columns=features.columns)
    return passenger_ids, scaled_features_df