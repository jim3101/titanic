import pandas as pd


class FeatureSet():

    def __init__(self):
        self.features = []

    def add_feature(self, feature):
        self.features.append(feature)

    def add_features(self, features):
        for feature in features:
            self.add_feature(feature)

    def preprocess_features(self):
        for feature in self.features:
            feature.preprocess()

    def get_features(self):
        features_df = pd.DataFrame()
        for feature in self.features:
            features_df[feature.column_name] = feature.get_feature()

        return features_df