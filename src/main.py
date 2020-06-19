from preprocess.preprocess_data import DataPreprocessor
from model import Model


def main():
    train_data_preprocessor = DataPreprocessor(dataset='train')
    train_data_preprocessor.preprocess()

    test_data_preprocessor = DataPreprocessor(dataset='test', scaler=train_data_preprocessor.get_scaler())
    test_data_preprocessor.preprocess()

    svc_model = Model()
    svc_model.fit(train_data_preprocessor)
    svc_model.predict(test_data_preprocessor)

if __name__ == '__main__':
    main()