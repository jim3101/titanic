from sklearn import svm
from sklearn.model_selection import cross_val_score
import pandas as pd
from constants import RESULTS_PATH


class Model():

    def __init__(self):
        self.model = svm.SVC(kernel='poly')

    def fit(self, train_data_preprocessor):
        features = train_data_preprocessor.get_features()
        target = train_data_preprocessor.get_target()
        self.model.fit(features, target)

    def predict(self, test_data_preprocessor):
        test_features = test_data_preprocessor.get_features()
        predictions = self.model.predict(test_features)
        passenger_ids = test_data_preprocessor.get_passenger_ids()
        self.store_results(passenger_ids, predictions)

    @staticmethod
    def store_results(passenger_ids, predictions):
        results = pd.DataFrame(passenger_ids)
        results['Survived'] = predictions
        results.to_csv(RESULTS_PATH, index=False)