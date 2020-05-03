from preprocess_data import load_raw_data, preprocess_raw_data


def main():
    raw_data = load_raw_data('data/train.csv')
    data = preprocess_raw_data(raw_data)

if __name__ == '__main__':
    main()