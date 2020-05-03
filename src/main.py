from preprocess_data import load_raw_data, get_train_data, get_test_data
from model import fit_model, make_predictions, evaluate_model
from results import save_predictions_as_csv


PATH_TO_TRAIN_DATA = 'data/train.csv'
PATH_TO_TEST_DATA = 'data/test.csv'

def main():
    train_features, target, scaler = get_train_data(PATH_TO_TRAIN_DATA)
    evaluate_model(train_features, target)
    model = fit_model(train_features, target)

    passenger_ids, test_features = get_test_data(PATH_TO_TEST_DATA, scaler)
    predictions = make_predictions(model, test_features)
    save_predictions_as_csv(passenger_ids, predictions)

if __name__ == '__main__':
    main()