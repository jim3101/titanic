import pandas as pd
import numpy as np
from constants import FEMALE_ENCODING, MALE_ENCODING, PORTS_ENCODING


class DataClass():

    def __init__(self, data, column_name):
        self.data = data
        self.column_name = column_name
        self.feature = pd.DataFrame({column_name: []})

    def preprocess(self):
        raise NotImplementedError()

    def get_feature(self):
        return self.feature


class GenderData(DataClass):
    def preprocess(self):
        self.feature = self.data[self.column_name].copy()
        self.feature.loc[self.feature == 'female'] = FEMALE_ENCODING
        self.feature.loc[self.feature == 'male'] = MALE_ENCODING
        self.feature = self.feature.apply(pd.to_numeric)


class AgeData(DataClass):
    def preprocess(self):
        self.feature = self.data[self.column_name].copy()
        # Replace NaNs in Age column by the average age
        self.feature.loc[pd.isna(self.feature)] = self.feature.mean()


class FamilyData(DataClass):
    def preprocess(self):
        self.feature = self.data['SibSp'] + self.data['Parch']


class FareData(DataClass):
    def preprocess(self):
        self.feature = self.data[self.column_name].copy()
        # Replace NaNs in Fare column by the average fare for the class
        for index, row in self.data.iterrows():
            if pd.isna(row['Fare']):
                average_fare_for_class = np.mean([x['Fare'] for i, x in self.data.iterrows()
                                                  if x['Pclass'] == row['Pclass'] and not pd.isna(x['Fare'])])
                self.feature.loc[index] = average_fare_for_class


class CabinData(DataClass):
    def preprocess(self):
        self.feature = self.data[self.column_name].copy()

        # Keep only the first letter from the cabin data
        self.feature.loc[:] = [x[0] if not pd.isna(x) else 0 for x in self.feature]

        # Make the letters numeric (A->1, B->2, ...,  no entry->0)
        self.feature.loc[:] = [ord(x) - 64 if not x == '0' else '0' for x in self.feature]
        self.feature = pd.to_numeric(self.feature)


class EmbarkedData(DataClass):
    def preprocess(self):
        self.feature = self.data[self.column_name].copy()
        self.feature.loc[:] = [PORTS_ENCODING[x] if x in PORTS_ENCODING else len(PORTS_ENCODING) for x in self.feature]
        self.feature = pd.to_numeric(self.feature)