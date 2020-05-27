import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from constants import PATH_TO_TRAIN_DATA, PATH_TO_TEST_DATA, FEMALE_ENCODING, MALE_ENCODING, PORTS_ENCODING


class DataPreprocessor():

    def __init__(self, dataset='train', scaler=None):
        if dataset == 'train':
            self.data = self.load_raw_data(PATH_TO_TRAIN_DATA)
        elif dataset == 'test':
            self.data = self.load_raw_data(PATH_TO_TEST_DATA)

        self.dataset = dataset
        self.scaler = scaler
        self.features = None
        self.target = None
        self.passenger_ids = None
        self.preprocessing_done = False

    @staticmethod
    def load_raw_data(path_to_csv):
        return pd.read_csv(path_to_csv)

    def preprocess(self):
        self.data = self.data.drop(['Name', 'Ticket'], axis='columns')
        if self.dataset != 'test':
            self.data = self.data.drop('PassengerId', axis='columns')
        self.preprocess_all_columns()
        self.split_features_target_ids()
        self.scale_features()
        self.preprocessing_done = True

    def get_features(self):
        assert self.preprocessing_done == True
        return self.features

    def get_target(self):
        assert self.preprocessing_done == True
        return self.target

    def get_passenger_ids(self):
        assert self.preprocessing_done == True
        return self.passenger_ids

    def preprocess_all_columns(self):
        # Preprocess columns that require preprocessing
        self.preprocess_gender_data()
        self.preprocess_age_data()
        self.preprocess_family_data()
        self.preprocess_fare_data()
        self.preprocess_cabin_data()
        self.preprocess_embarked_data()

    def split_features_target_ids(self):
        if self.dataset == 'train':
            self.target = self.data['Survived']
            self.features = self.data.drop('Survived', axis='columns')
        elif self.dataset == 'test':
            self.passenger_ids = self.data['PassengerId']
            self.features = self.data.drop('PassengerId', axis='columns')

    def scale_features(self):
        if self.scaler is None:
            self.scaler = MinMaxScaler()
            self.scaler.fit(self.features)

        scaled_features = self.scaler.transform(self.features)
        scaled_features_df = pd.DataFrame(data=scaled_features, columns=self.features.columns)
        self.features = scaled_features_df

    def get_scaler(self):
        return self.scaler

    def preprocess_gender_data(self):
        self.data.loc[self.data['Sex'] == 'female', 'Sex'] = FEMALE_ENCODING
        self.data.loc[self.data['Sex'] == 'male', 'Sex'] = MALE_ENCODING
        self.data['Sex'] = self.data['Sex'].apply(pd.to_numeric)

    def preprocess_age_data(self):
        # Replace NaNs in Age column by the average age
        self.data.loc[pd.isna(self.data['Age']), 'Age'] = self.data['Age'].mean()

    def preprocess_family_data(self):
        self.data['Family'] = self.data['SibSp'] + self.data['Parch']
        self.data = self.data.drop(['SibSp', 'Parch'], axis='columns')

    def preprocess_fare_data(self):
        # Replace NaNs in Fare column by the average fare for the class
        for index, row in self.data.iterrows():
            if pd.isna(row['Fare']):
                average_fare_for_class = np.mean([x['Fare'] for i, x in self.data.iterrows() \
                                                  if x['Pclass'] == row['Pclass'] and not pd.isna(x['Fare'])])
                self.data.loc[index, 'Fare'] = average_fare_for_class

    def preprocess_cabin_data(self):
        # Keep only the first letter from the cabin data
        self.data['Cabin'] = [x[0] if not pd.isna(x) else 0 for x in self.data['Cabin']]
        # Make the letters numeric (A->1, B->2, ...,  no entry->0)
        self.data['Cabin'] = [ord(x) - 64 if not x == 0 else 0 for x in self.data['Cabin']]
        self.data['Cabin'] = self.data['Cabin'].apply(pd.to_numeric)

    def preprocess_embarked_data(self):
        self.data['Embarked'] = [PORTS_ENCODING[x] if x in PORTS_ENCODING else len(PORTS_ENCODING) \
                                 for x in self.data['Embarked']]
        self.data['Embarked'] = self.data['Embarked'].apply(pd.to_numeric)