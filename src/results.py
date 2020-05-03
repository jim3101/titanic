import pandas as pd


RESULTS_PATH = 'data/results.csv'

def save_predictions_as_csv(passenger_ids, predictions):
    results = pd.DataFrame(passenger_ids)
    results['Survived'] = predictions
    results.to_csv(RESULTS_PATH, index=False)