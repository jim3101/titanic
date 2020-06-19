import pandas as pd
# import numpy as np
from sklearn.preprocessing import MinMaxScaler
from preprocess.feature_classes import GenderData, AgeData, FamilyData, FareData, CabinData, EmbarkedData
from preprocess.feature_set import FeatureSet
from constants import PATH_TO_TRAIN_DATA, PATH_TO_TEST_DATA


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

    def preprocess(self):
        self.data = self.data.drop(['Name', 'Ticket'], axis='columns')
        if self.dataset != 'test':
            self.data = self.data.drop('PassengerId', axis='columns')

        self.split_features_target_ids()
        self.preprocess_features()
        self.scale_features()

        self.preprocessing_done = True

    def split_features_target_ids(self):
        if self.dataset == 'train':
            self.target = self.data['Survived']
            self.features = self.data.drop('Survived', axis='columns')
        elif self.dataset == 'test':
            self.passenger_ids = self.data['PassengerId']
            self.features = self.data.drop('PassengerId', axis='columns')

    def preprocess_features(self):
        gender_data = GenderData(self.features, 'Sex')
        age_data = AgeData(self.features, 'Age')
        family_data = FamilyData(self.features, 'Family')
        fare_data = FareData(self.features, 'Fare')
        cabin_data = CabinData(self.features, 'Cabin')
        embarked_data = EmbarkedData(self.features, 'Embarked')

        feature_set = FeatureSet()
        feature_set.add_features([gender_data, age_data, family_data, fare_data, cabin_data, embarked_data])
        feature_set.preprocess_features()
        self.features = feature_set.get_features()

    def scale_features(self):
        if self.scaler is None:
            self.scaler = MinMaxScaler()
            self.scaler.fit(self.features)

        scaled_features = self.scaler.transform(self.features)
        scaled_features_df = pd.DataFrame(data=scaled_features, columns=self.features.columns)
        self.features = scaled_features_df

    @staticmethod
    def load_raw_data(path_to_csv):
        return pd.read_csv(path_to_csv)

    def get_features(self):
        assert self.preprocessing_done == True
        return self.features

    def get_target(self):
        assert self.preprocessing_done == True
        return self.target

    def get_passenger_ids(self):
        assert self.preprocessing_done == True
        return self.passenger_ids

    def get_scaler(self):
        return self.scaler
